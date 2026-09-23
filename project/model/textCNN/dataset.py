"""
從 pcap_process/for_datasets/pcap_label.py 標記好的 flow 直接串流出 textCNN
訓練用的 (token_ids, label_id)，不落地存檔——用完即丟，訓練資料集再大也不會
額外佔硬碟空間，只吃處理過程中的記憶體（shuffle buffer 的大小）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset

try:  # 以套件方式執行
    from .tokenizer import ByteTokenizer
    from ..pcap_process.for_datasets.pcap_label import iter_labeled_flows
except ImportError:  # 直接執行
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from model.textCNN.tokenizer import ByteTokenizer
    from pcap_process.for_datasets.pcap_label import iter_labeled_flows

ATTEMPTED_SUFFIX = " - Attempted"

# Feb 22 / Feb 23 兩天都是同一組 Web 應用層攻擊（Brute Force -Web / -XSS /
# SQL Injection），標記後的 Label 就是這幾種 + BENIGN + 各自的 "- Attempted"
# 變形。這裡先寫死，兩份背景標記工作跑完後應該要跟實際分佈對一次帳。
DEFAULT_CLASSES = [
    "BENIGN",
    "Web Attack - Brute Force",
    "Web Attack - SQL",
    "Web Attack - XSS",
]


def normalize_label(label: str | None) -> str | None:
    """textCNN 只看 payload 內容。"X - Attempted"（攻擊沒有真的成功）在
    payload 內容上（攻擊者送出的注入字串本身）跟成功的 "X" 幾乎沒有差別——
    差別在於伺服器端有沒有真的中招，這件事單看 client 送出的 request 通常看
    不出來，所以併回父類別 X。這跟 LightGBM 那邊考慮併回 BENIGN 是不同的理
    由（那邊在乎的是流量統計特徵，Attempted 的封包數/時長確實比較短）。
    label 是 None（pcap 裡的 flow 沒能對回 CSV 任何一列）保留 None，呼叫端
    自己決定要不要丟掉這筆樣本（訓練時應該丟掉，因為不知道真正的 label）。
    """
    if label is None:
        return None
    if label.endswith(ATTEMPTED_SUFFIX):
        return label[: -len(ATTEMPTED_SUFFIX)]
    return label


@dataclass
class DayShard:
    """一天份的 pcap 目錄 + 對應的已修正 CSV。"""
    pcap_dir: str
    csv_path: str
    calibration_pcap: str | None = None


class LabeledPcapDataset(IterableDataset):
    """對每個 DayShard 依序呼叫 pcap_label.iter_labeled_flows()，把 payload
    轉成 ByteTokenizer 的 id 序列、label 轉成 class id，用一個固定大小的
    shuffle buffer 做局部打亂後串流吐出去（無法像存在記憶體裡的 Dataset 那樣
    整份打亂，buffer 大小換取隨機性，不是資料集大小）。

    多 worker（DataLoader num_workers>0）時用「檔案層級」分工，每個 worker
    處理不同子集的 pcap_dir，避免重複讀取。
    """

    def __init__(
        self,
        shards: list[DayShard],
        classes: list[str] = DEFAULT_CLASSES,
        max_len: int = 1500,
        shuffle_buffer: int = 2000,
        seed: int = 42,
        max_flows: int | None = None,
        minority_replay_every: int | None = None,
    ):
        self.shards = shards
        self.classes = classes
        self.class_to_id = {c: i for i, c in enumerate(classes)}
        self.max_len = max_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_flows = max_flows  # 只用來快速跑小規模驗證，正式訓練留 None
        # 見檔案開頭的說明：攻擊流量在 pcap 檔名排序下會整批擠在串流最前面
        # （受害伺服器那份 UCAP*.pcap 檔名的 "U" 排在 "cap*" 的 "c" 前面），
        # 单靠一個 shuffle_buffer 沒辦法把它跟後面幾百萬筆 BENIGN 混在一起。
        # 這裡另外快取看過的少數類樣本，每看過 N 筆多數類（BENIGN）樣本就
        # 重播一次快取裡隨機一筆，讓少數類訊號整個 epoch 都持續出現，不是
        # None（不重播）就是隔多久重播一次。評估／驗證資料集要留 None，不
        # 然驗證出來的分佈就不是真實分佈了。
        self.minority_replay_every = minority_replay_every

    def _labeled_samples(self):
        tokenizer = ByteTokenizer(max_len=self.max_len)
        rng = random.Random(self.seed)
        benign_id = self.class_to_id.get("BENIGN")
        minority_cache: list[tuple[torch.Tensor, int]] = []
        since_replay = 0
        n = 0
        for shard in self.shards:
            for lf in iter_labeled_flows(shard.pcap_dir, shard.csv_path,
                                          calibration_pcap=shard.calibration_pcap):
                label = normalize_label(lf.label)
                if label is None or label not in self.class_to_id:
                    continue
                cid = self.class_to_id[label]
                ids = torch.tensor(tokenizer.encode(lf.payload), dtype=torch.long)
                item = (ids, cid)
                yield item
                n += 1
                if self.max_flows is not None and n >= self.max_flows:
                    return

                if self.minority_replay_every is None:
                    continue
                if cid != benign_id:
                    minority_cache.append(item)
                else:
                    since_replay += 1
                    if minority_cache and since_replay >= self.minority_replay_every:
                        since_replay = 0
                        yield minority_cache[rng.randrange(len(minority_cache))]
                        n += 1
                        if self.max_flows is not None and n >= self.max_flows:
                            return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None and worker_info.num_workers > 1:
            shards = shards[worker_info.id::worker_info.num_workers]
        orig_shards, self.shards = self.shards, shards
        try:
            rng = random.Random(self.seed)
            buf = []
            for item in self._labeled_samples():
                if len(buf) < self.shuffle_buffer:
                    buf.append(item)
                    continue
                idx = rng.randrange(len(buf))
                yield buf[idx]
                buf[idx] = item
            rng.shuffle(buf)
            yield from buf
        finally:
            self.shards = orig_shards
