"""
以 CSE-CIC-IDS2018 的 CSV 流量統計特徵（CICFlowMeter 欄位）訓練 / 推論用的
LightGBM 分類器。

跟 textCNN（吃 flow payload 的 raw bytes）不同，這裡吃的是每條 flow 的統計
特徵（Flow Duration、Total Fwd Packet、Flow Bytes/s ... 等，欄位名稱依
csv/ 底下實際的 CICFlowMeter fork 輸出而定，不是官方精簡過的欄位名），最後
一欄 Label 是分類目標（BENIGN / 各攻擊名稱；dataset.py 已把誤標的攻擊列修
正過）。CSV 讀取／欄位對齊邏輯統一放在
pcap_process/for_datasets/for_csv.py，這裡只呼叫。

CLI 用法（於專案根目錄執行）：
    python -m model.LightGBM.model train                # 用 csv/ 底下所有 csv 訓練並存檔
    python -m model.LightGBM.model train a.csv b.csv     # 指定要用哪幾份 csv
    python -m model.LightGBM.model eval some_day.csv     # 載入已存好的模型並評估

當函式庫用（例如 pcap_process/online.py 的下游 handler）：
    from model.LightGBM.model import LightGBMIDS
    clf = LightGBMIDS.load()
    label = clf.predict_one({"Flow Duration": 123.0, "Total Fwd Packet": 4, ...})
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import lightgbm as lgb

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(DIR))  # model/LightGBM -> model -> 專案根目錄

try:  # 以套件方式執行： python -m model.LightGBM.model
    from pcap_process.for_datasets.for_csv import CSV_DIR, default_csv_paths, load_csv, load_dataset
except ImportError:  # 直接執行： python model/LightGBM/model.py
    import sys
    sys.path.insert(0, ROOT_DIR)
    from pcap_process.for_datasets.for_csv import CSV_DIR, default_csv_paths, load_csv, load_dataset

DEFAULT_MODEL_PATH = os.path.join(DIR, "ids_lightgbm.txt")
DEFAULT_META_PATH = os.path.join(DIR, "ids_lightgbm.meta.json")


ATTEMPTED_SUFFIX = " - Attempted"


def normalize_labels(y_labels: list[str], *, merge_attempted: bool = True) -> list[str]:
    """把 "X - Attempted"（連線嘗試但沒真的成功）視為無害流量，併回 BENIGN。

    CICFlowMeter fork 會把失敗/未完成的攻擊連線另外標成 "X - Attempted"。
    如果不合併，模型要多學好幾個「幾乎跟 BENIGN 長一樣、只差在連線沒打成功」
    的類別；併回 BENIGN 後類別數變少，且更貼近「這條流量有沒有造成危害」的
    實際判斷需求。注意：某些攻擊在這批 CSV 裡只出現過 Attempted 版本（例如
    FTP-BruteForce），合併後該類別會整個消失，模型就學不到它。
    """
    if not merge_attempted:
        return y_labels
    return [
        "BENIGN" if label.endswith(ATTEMPTED_SUFFIX) else label
        for label in y_labels
    ]


def build_label_map(y_labels: list[str]) -> list[str]:
    """建立 class id -> label 名稱的對照表。BENIGN（若存在）固定排在 id 0。"""
    uniq = sorted(set(y_labels))
    if "BENIGN" in uniq:
        uniq.remove("BENIGN")
        uniq = ["BENIGN"] + uniq
    return uniq


def encode_labels(y_labels: list[str], classes: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(classes)}
    return np.asarray([idx[label] for label in y_labels], dtype=np.int32)


@dataclass
class LightGBMIDS:
    """封裝 LightGBM booster + feature/label 對照表，統一存檔 / 讀檔 / 推論介面。"""

    booster: lgb.Booster | None = None
    feature_names: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)  # class id -> label 名稱

    @property
    def num_class(self) -> int:
        return len(self.classes)

    def fit(
        self,
        X: np.ndarray,
        y_ids: np.ndarray,
        *,
        valid_fraction: float = 0.1,
        seed: int = 42,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 30,
        params: dict | None = None,
    ) -> "LightGBMIDS":
        if not self.feature_names:
            raise ValueError("feature_names 尚未設定")
        if not self.classes:
            raise ValueError("classes 尚未設定")

        n = len(y_ids)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_valid = int(n * valid_fraction) if valid_fraction > 0 else 0
        valid_idx, train_idx = perm[:n_valid], perm[n_valid:]

        train_set = lgb.Dataset(
            X[train_idx], label=y_ids[train_idx], feature_name=self.feature_names,
        )
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=50)]
        if n_valid:
            valid_set = lgb.Dataset(
                X[valid_idx], label=y_ids[valid_idx],
                feature_name=self.feature_names, reference=train_set,
            )
            valid_sets.append(valid_set)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        train_params = {
            "objective": "multiclass",
            "num_class": self.num_class,
            "metric": "multi_logloss",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": seed,
        }
        if params:
            train_params.update(params)

        self.booster = lgb.train(
            train_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("模型尚未訓練 / 載入")
        X = np.asarray(X, dtype=np.float64)
        num_iteration = getattr(self.booster, "best_iteration", None) or None
        return self.booster.predict(X, num_iteration=num_iteration)

    def predict(self, X) -> list[str]:
        proba = self.predict_proba(X)
        ids = proba.argmax(axis=1)
        return [self.classes[i] for i in ids]

    def predict_one(self, features: dict | list | np.ndarray) -> str:
        """單一 flow 的推論。features 可以是 {欄位名: 數值} 的 dict，
        也可以是依 self.feature_names 順序排好的 list/array。"""
        if isinstance(features, dict):
            row = [features.get(name, np.nan) for name in self.feature_names]
        else:
            row = list(features)
        return self.predict([row])[0]

    def save(self, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH):
        if self.booster is None:
            raise RuntimeError("模型尚未訓練，無法儲存")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        # 不能直接呼叫 booster.save_model(path) 讓 LightGBM 的 C API 開檔：
        # 專案路徑含中文字（.../專題/...），LightGBM 在 Windows 上底層用 C 的
        # fopen 開檔，是用系統 ANSI code page 解碼路徑而非 UTF-8，中文路徑會
        # 被解成無效路徑，導致 "Model file ... is not available for writes"。
        # 改用 model_to_string() 取出模型內容，用 Python 自己的檔案 IO（原生
        # 支援 Windows 寬字元路徑）寫檔，完全避開這個問題。
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(self.booster.model_to_string())
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"feature_names": self.feature_names, "classes": self.classes},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH) -> "LightGBMIDS":
        with open(model_path, "r", encoding="utf-8") as f:
            booster = lgb.Booster(model_str=f.read())
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(booster=booster, feature_names=meta["feature_names"], classes=meta["classes"])


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> str:
    """不依賴 scikit-learn 的簡易分類報告（precision / recall / f1 / support）。"""
    stats = defaultdict(lambda: [0, 0, 0])  # class id -> [tp, fp, fn]
    for t, p in zip(y_true, y_pred):
        t, p = int(t), int(p)
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

    correct = int((y_true == y_pred).sum())
    lines.append(f"\naccuracy: {correct / len(y_true):.4f} ({correct}/{len(y_true)})")
    return "\n".join(lines)


def train(
    csv_paths: list[str] | None = None,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    meta_path: str = DEFAULT_META_PATH,
    valid_fraction: float = 0.1,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    seed: int = 42,
    merge_attempted: bool = True,
) -> LightGBMIDS:
    csv_paths = csv_paths or default_csv_paths()
    if not csv_paths:
        raise FileNotFoundError(f"{CSV_DIR} 底下找不到任何 csv，請指定 --csv")

    print(f"讀取 {len(csv_paths)} 份 CSV: {[os.path.basename(p) for p in csv_paths]}")
    X, y_labels, feature_names = load_dataset(csv_paths)
    y_labels = normalize_labels(y_labels, merge_attempted=merge_attempted)
    classes = build_label_map(y_labels)
    y_ids = encode_labels(y_labels, classes)
    print(f"樣本數: {len(y_ids)}, 特徵數: {len(feature_names)}, 類別: {classes}")

    clf = LightGBMIDS(feature_names=feature_names, classes=classes)
    clf.fit(
        X, y_ids,
        valid_fraction=valid_fraction,
        seed=seed,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )
    clf.save(model_path, meta_path)
    print(f"模型已存到 {model_path}\n特徵/類別資訊已存到 {meta_path}")
    return clf


def evaluate(
    csv_paths: list[str],
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    meta_path: str = DEFAULT_META_PATH,
    merge_attempted: bool = True,
) -> None:
    clf = LightGBMIDS.load(model_path, meta_path)

    parts = [load_csv(p) for p in csv_paths]
    col_of_list = [{name: i for i, name in enumerate(names)} for _, _, names in parts]
    X_list, y_list = [], []
    for (X, y, _), col_of in zip(parts, col_of_list):
        reorder = [col_of[n] for n in clf.feature_names]
        X_list.append(X[:, reorder])
        y_list.extend(y)
    X = np.concatenate(X_list, axis=0)
    y_list = normalize_labels(y_list, merge_attempted=merge_attempted)

    unknown = sorted(set(y_list) - set(clf.classes))
    if unknown:
        print(f"[警告] 評估資料含模型沒看過的類別，將被忽略: {unknown}")
    label_to_id = {name: i for i, name in enumerate(clf.classes)}
    keep = [i for i, label in enumerate(y_list) if label in label_to_id]
    X = X[keep]
    y_ids = np.asarray([label_to_id[y_list[i]] for i in keep], dtype=np.int32)

    proba = clf.predict_proba(X)
    pred_ids = proba.argmax(axis=1)
    print(classification_report(y_ids, pred_ids, clf.classes))


def main():
    ap = argparse.ArgumentParser(description="CSE-CIC-IDS2018 LightGBM 分類器")
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="訓練並儲存模型")
    p_train.add_argument("csv", nargs="*", help="訓練用的 csv 路徑，省略則用 csv/ 底下所有 csv")
    p_train.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--meta", default=DEFAULT_META_PATH)
    p_train.add_argument("--valid-fraction", type=float, default=0.1)
    p_train.add_argument("--num-boost-round", type=int, default=500)
    p_train.add_argument("--early-stopping-rounds", type=int, default=30)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--keep-attempted", action="store_true",
                          help="不要把 'X - Attempted' 併入 BENIGN，保留成獨立類別")

    p_eval = sub.add_parser("eval", help="載入已存好的模型並評估")
    p_eval.add_argument("csv", nargs="+", help="評估用的 csv 路徑")
    p_eval.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p_eval.add_argument("--meta", default=DEFAULT_META_PATH)
    p_eval.add_argument("--keep-attempted", action="store_true",
                         help="不要把 'X - Attempted' 併入 BENIGN（須與訓練時的設定一致）")

    args = ap.parse_args()

    if args.command == "train":
        train(
            args.csv or None,
            model_path=args.model,
            meta_path=args.meta,
            valid_fraction=args.valid_fraction,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            seed=args.seed,
            merge_attempted=not args.keep_attempted,
        )
    elif args.command == "eval":
        evaluate(args.csv, model_path=args.model, meta_path=args.meta,
                 merge_attempted=not args.keep_attempted)


if __name__ == "__main__":
    main()
