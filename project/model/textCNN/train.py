"""
用 pcap + 已修正的 CICFlowMeter CSV（見 pcap_process/for_datasets/pcap_label.py）
預訓練 textCNN。

跟 LightGBM 的 model.py 同樣的精神（CLI 用法、存檔格式盡量一致），但資料來
源是直接串流讀 pcap，不會把整個資料集攤開存成一份新檔案／預先載進記憶體
——所以沒辦法像 LightGBM 那樣「先讀全部再隨機切 train/valid」。這裡改用
「照擷取日期切」：一天當訓練集，另一天當驗證集，兩天是分開擷取的獨立資料，
不會有同一條 flow 同時出現在兩邊的 leak 問題。

因為是串流讀 pcap，每個 epoch 都要重新讀一次整份 pcap 語料（不是從記憶體裡
的資料集重複取樣），目前實測單執行緒約 29.5MB/s，如果 train+valid 合計
~130GB，一個 epoch 大概要 70~90 分鐘——這是這個資料規模下用串流方式訓練必然
的代價，多 epoch 訓練會很貴，預設只跑 1 epoch（樣本數本身有上百萬筆，對這
個只有 ~19 萬參數的小模型來說單一 epoch 的梯度更新次數通常已經足夠，不是
因為怕麻煩才省略多 epoch）。

CLI 用法：
    python -m model.textCNN.train \\
        --train-pcap pcap/pcap2-22 --train-csv pcap/pcap2-22/Thursday-22-02-2018.csv \\
        --valid-pcap pcap/pcap2-23 --valid-csv pcap/pcap2-23/Friday-23-02-2018.csv

先用 --max-flows 幾千筆跑一次快速驗證整條管線沒問題，再拿掉這個限制跑正式
訓練。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:  # 以套件方式執行： python -m model.textCNN.train
    from .dataset import DEFAULT_CLASSES, DayShard, LabeledPcapDataset, normalize_label
    from .embedding import ByteEmbedding
    from .model import TextCNN
    from .tokenizer import ByteTokenizer
except ImportError:  # 直接執行
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from model.textCNN.dataset import DEFAULT_CLASSES, DayShard, LabeledPcapDataset, normalize_label
    from model.textCNN.embedding import ByteEmbedding
    from model.textCNN.model import TextCNN
    from model.textCNN.tokenizer import ByteTokenizer

DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(DIR, "ids_textcnn.pt")
DEFAULT_META_PATH = os.path.join(DIR, "ids_textcnn.meta.json")

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TextCNNIDS:
    """封裝 ByteEmbedding + TextCNN + class 對照表，統一存檔／讀檔／推論介面
    （跟 model/LightGBM/model.py 的 LightGBMIDS 是同樣的設計）。"""

    embedding: ByteEmbedding
    model: TextCNN
    classes: list[str] = field(default_factory=list)
    max_len: int = 1500

    def predict_proba(self, ids: torch.Tensor) -> torch.Tensor:
        self.embedding.eval()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.embedding(ids))
            return torch.softmax(logits, dim=1)

    def predict(self, ids: torch.Tensor) -> list[str]:
        proba = self.predict_proba(ids)
        idx = proba.argmax(dim=1).tolist()
        return [self.classes[i] for i in idx]

    def predict_one(self, payload: bytes) -> str:
        """單一 flow payload（bytes）的推論。"""
        tokenizer = ByteTokenizer(max_len=self.max_len)
        ids = tokenizer(payload).unsqueeze(0)
        return self.predict(ids)[0]

    def save(self, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save(
            {"embedding": self.embedding.state_dict(), "model": self.model.state_dict()},
            model_path,
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"classes": self.classes, "max_len": self.max_len,
                 "embed_dim": self.embedding.embed_dim},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH) -> "TextCNNIDS":
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        embedding = ByteEmbedding(embed_dim=meta["embed_dim"])
        model = TextCNN(embed_dim=meta["embed_dim"], num_classes=len(meta["classes"]))
        ckpt = torch.load(model_path, map_location="cpu")
        embedding.load_state_dict(ckpt["embedding"])
        model.load_state_dict(ckpt["model"])
        return cls(embedding=embedding, model=model, classes=meta["classes"], max_len=meta["max_len"])


def class_weights_from_manifests(manifest_paths: list[str], classes: list[str]) -> torch.Tensor:
    """從 pcap_label.py --manifest 的輸出統計各類別（套用跟 dataset.py 一樣的
    normalize_label 合併 "- Attempted"）的樣本數，算類別權重（standard
    "balanced" 公式：total/(num_classes*count)）。

    實測 BENIGN 對單一攻擊類別的比例高達 1:14000~1:18000，CrossEntropyLoss
    不加權重的話，模型幾乎確定會學成「永遠猜 BENIGN」（畢竟這樣就有
    99.99%+ 的訓練準確率），加權重讓少數的攻擊樣本在 loss 裡的份量不會被
    多數的 BENIGN 徹底淹沒。
    """
    class_to_id = {c: i for i, c in enumerate(classes)}
    counts = Counter()
    for path in manifest_paths:
        for chunk in pd.read_csv(path, usecols=["label"], chunksize=500_000):
            for raw in chunk["label"].dropna():
                label = normalize_label(raw)
                if label in class_to_id:
                    counts[label] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("manifest 裡沒有任何屬於 classes 的樣本，無法算權重")
    n_classes = len(classes)
    weights = [total / (n_classes * counts.get(c, 1)) for c in classes]
    print(f"[class_weights] 樣本數={dict(counts)}  weights={[round(w, 2) for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


def _classification_report(y_true: list[int], y_pred: list[int], classes: list[str]) -> str:
    """不依賴 scikit-learn 的簡易分類報告，跟 LightGBM 那邊的風格一致。"""
    stats = defaultdict(lambda: [0, 0, 0])  # class id -> [tp, fp, fn]
    for t, p in zip(y_true, y_pred):
        if t == p:
            stats[t][0] += 1
        else:
            stats[p][1] += 1
            stats[t][2] += 1

    lines = [f"{'label':30s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>10s}"]
    for cid, name in enumerate(classes):
        tp, fp, fn = stats[cid]
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        lines.append(f"{name:30s} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}")

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    lines.append(f"\naccuracy: {correct / total:.4f} ({correct}/{total})" if total else "\n(無樣本)")
    return "\n".join(lines)


def evaluate(clf: TextCNNIDS, shards: list[DayShard], *, batch_size: int = 64,
             max_flows: int | None = None, device: str = DEFAULT_DEVICE) -> None:
    ds = LabeledPcapDataset(shards, classes=clf.classes, max_len=clf.max_len,
                             shuffle_buffer=1, max_flows=max_flows)
    loader = DataLoader(ds, batch_size=batch_size)

    clf.embedding.to(device)
    clf.model.to(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    for ids, labels in loader:
        proba = clf.predict_proba(ids.to(device))
        y_pred.extend(proba.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())
    clf.embedding.cpu()
    clf.model.cpu()

    print(_classification_report(y_true, y_pred, clf.classes))


def train(
    train_shards: list[DayShard],
    valid_shards: list[DayShard] | None = None,
    *,
    classes: list[str] = DEFAULT_CLASSES,
    embed_dim: int = 128,
    max_len: int = 1500,
    batch_size: int = 64,
    epochs: int = 1,
    lr: float = 1e-3,
    shuffle_buffer: int = 2000,
    max_flows: int | None = None,
    log_every: int = 200,
    device: str = DEFAULT_DEVICE,
    class_weights: torch.Tensor | None = None,
    minority_replay_every: int | None = 200,
) -> TextCNNIDS:
    embedding = ByteEmbedding(embed_dim=embed_dim).to(device)
    net = TextCNN(embed_dim=embed_dim, num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(list(embedding.parameters()) + list(net.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    train_ds = LabeledPcapDataset(train_shards, classes=classes, max_len=max_len,
                                   shuffle_buffer=shuffle_buffer, max_flows=max_flows,
                                   minority_replay_every=minority_replay_every)
    loader = DataLoader(train_ds, batch_size=batch_size)

    for epoch in range(epochs):
        embedding.train()
        net.train()
        running_loss, seen, correct = 0.0, 0, 0
        class_seen = Counter()
        for step, (ids, labels) in enumerate(loader):
            ids, labels = ids.to(device), labels.to(device)
            opt.zero_grad()
            logits = net(embedding(ids))
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            net.apply_max_norm()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            seen += bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            class_seen.update(labels.cpu().tolist())

            if (step + 1) % log_every == 0:
                dist = {classes[i]: n for i, n in sorted(class_seen.items())}
                print(f"[epoch {epoch}] step {step + 1:>6d}  "
                      f"loss={running_loss / seen:.4f}  acc={correct / seen:.4f}  "
                      f"seen={seen}  類別分佈={dist}")

        print(f"epoch {epoch} 完成：loss={running_loss / max(seen, 1):.4f} "
              f"acc={correct / max(seen, 1):.4f}，共 {seen} 筆樣本")

    clf = TextCNNIDS(embedding=embedding.cpu(), model=net.cpu(), classes=classes, max_len=max_len)

    if valid_shards:
        print("\n=== 驗證集（另一天的資料）===")
        evaluate(clf, valid_shards, batch_size=batch_size, max_flows=max_flows, device=device)

    return clf


def main():
    ap = argparse.ArgumentParser(description="textCNN 預訓練（pcap + 已修正 CICFlowMeter CSV）")
    ap.add_argument("--train-pcap", required=True, help="訓練用的一天份 pcap 資料夾")
    ap.add_argument("--train-csv", required=True, help="對應的已修正 CICFlowMeter CSV")
    ap.add_argument("--valid-pcap", default=None, help="驗證用的另一天 pcap 資料夾（省略則不驗證）")
    ap.add_argument("--valid-csv", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=1500)
    ap.add_argument("--shuffle-buffer", type=int, default=2000)
    ap.add_argument("--max-flows", type=int, default=None,
                     help="只用前 N 筆樣本，跑快速驗證管線用；正式訓練請拿掉這個參數")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--meta", default=DEFAULT_META_PATH)
    ap.add_argument("--weight-manifest", nargs="+", default=None,
                     help="一或多個 pcap_label.py --manifest 輸出的 csv，用來算類別權重。"
                          "注意：預設已經用 --minority-replay-every 做過取樣層級的重播，"
                          "兩個一起開會疊加過頭（loss 權重是照原始 1:1萬多的比例算的），"
                          "通常兩個只需要開一個，不確定就先只用重播")
    ap.add_argument("--minority-replay-every", type=int, default=200,
                     help="訓練時每看過幾筆 BENIGN 樣本就重播一次少數類快取（見 dataset.py "
                          "的說明：攻擊流量在 pcap 檔名排序下會整批擠在串流最前面，沒有這個機制"
                          "模型在看完幾百萬筆純 BENIGN 之後會把攻擊特徵忘光）。設 0 或負數關閉")
    args = ap.parse_args()

    train_shards = [DayShard(args.train_pcap, args.train_csv)]
    valid_shards = [DayShard(args.valid_pcap, args.valid_csv)] if args.valid_pcap else []

    class_weights = (class_weights_from_manifests(args.weight_manifest, DEFAULT_CLASSES)
                      if args.weight_manifest else None)

    minority_replay_every = args.minority_replay_every if args.minority_replay_every > 0 else None

    print(f"device={args.device}  train={args.train_pcap}  valid={args.valid_pcap or '(無)'}  "
          f"minority_replay_every={minority_replay_every}")
    clf = train(
        train_shards, valid_shards,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        max_len=args.max_len, shuffle_buffer=args.shuffle_buffer,
        max_flows=args.max_flows, device=args.device, class_weights=class_weights,
        minority_replay_every=minority_replay_every,
    )
    clf.save(args.model, args.meta)
    print(f"模型已存到 {args.model}\n類別資訊已存到 {args.meta}")


if __name__ == "__main__":
    main()
