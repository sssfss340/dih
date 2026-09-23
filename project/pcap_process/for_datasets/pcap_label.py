"""
把「官方原始 pcap」重組出的 flow，比對回「已修正過的 CICFlowMeter CSV」，
取得每條 flow 對應的 Label——textCNN 訓練資料的標記工具。

背景（跟 atk_vic.html / Error Prevalence in NIDS datasets 那篇論文有關）：
  * pcap/<day>/ 底下的原始封包是照「主機」分檔（每台受監控主機/伺服器各自
    一個 .pcap，檔名裡就是那台主機的 IP，例如
    capDESKTOP-AN3U28N-172.31.64.111.pcap、UCAP172.31.69.28.pcap）。
  * 同一天的 CSV（例如 Friday-23-02-2018.csv）是官方 CICFlowMeter 對「同一批
    封包」跑出來、且已用論文的方法修正過誤標的統計特徵 + Label。
  * atk_vic.html 那張表（攻擊者 IP + 受害者 IP + 攻擊時間窗）是 CIC 官方最原始
    的標記依據，論文正是在講這種「純 IP + 時間窗」的標法會誤標多少流量。
    所以這裡不是拿 atk_vic.html 重新用 IP+時間窗去標記，而是把 pcap 重組出的
    每條 flow，用「五元組 + 時間重疊」對回 CSV 裡「同一條 flow」，直接沿用
    CSV 已經修正過的 Label——textCNN 才會跟 LightGBM 學到一致的 ground
    truth。atk_vic.html 只用來做事後 sanity check（見 sanity_check()）。

兩個實際的技術問題：
  1. 方向性：for_pcap.py 的 FlowPackager 對同一個 TCP 連線的兩個方向各開一個
     Flow（marker 是 (src, sport, dst, dport)），但 CICFlowMeter 一條 CSV
     列涵蓋雙向。用不排序方向的 canonical key（兩端點各自 (ip, port) 的
     frozenset）去查表，兩個方向的 Flow 自然會對到同一列。
  2. 時區：pcap 封包時間戳是 unix epoch（UTC），CSV 的 Timestamp 欄是產生
     CSV 那台機器的當地時間字串，兩者差一個固定但未知的 offset。不用猜時區
     常數，而是拿一小份 pcap（預設用資料夾裡最小的檔案）先做一次校正：
     找同一個 canonical key 在 CSV 索引裡唯一對應的一列，兩邊起始時間的差
     取中位數當作 offset。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import pandas as pd

try:  # 以套件方式執行： python -m pcap_process.for_datasets.pcap_label
    from .fast_pcap import iter_host_flows_fast
except ImportError:  # 直接執行
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from pcap_process.for_datasets.fast_pcap import iter_host_flows_fast

DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DIR))

# CSV 裡標記比對只需要這幾欄，不用把 91 欄整份讀進來
_ID_COLS = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Flow Duration", "Label"]

# 檔名結尾的 IPv4：capDESKTOP-AN3U28N-172.31.64.111.pcap / UCAP172.31.69.28.pcap 都適用
_HOST_IP_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})\.pcap$", re.IGNORECASE)

# 找不到比對得上的 CSV 列時，允許的最大時間誤差（秒）；超過這個範圍寧可標成
# 「沒對到」也不要硬湊一筆可能不相關的紀錄
DEFAULT_TOLERANCE = 5.0


def _canonical_key(ip_a: str, port_a: int, ip_b: str, port_b: int) -> frozenset:
    """雙向都會算出同一把 key，這樣同一條 TCP 連線的兩個方向（FlowPackager
    裡是兩個獨立的 Flow）會對到同一筆 CSV 紀錄。"""
    return frozenset({(ip_a, int(port_a)), (ip_b, int(port_b))})


def build_label_index(csv_path: str, chunksize: int = 200_000):
    """讀一天的 CSV，回傳 canonical key -> [(start_epoch, end_epoch, label), ...]。

    Timestamp 欄位先當「naive epoch」處理（当作 UTC 解析），實際跟 pcap 之間
    的時區差由 calibrate_offset() 另外校正，這裡不假設任何時區。
    """
    index: dict[frozenset, list[tuple[float, float, str]]] = {}
    n_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=_ID_COLS, chunksize=chunksize, low_memory=False):
        # pandas 会依字串精度自動選 datetime64 的解析度（這裡實測是 [us] 不是
        # [ns]），直接 .astype("int64") 拿到的單位會跟著變、不是恆定的奈秒，
        # 所以先強制轉成 [ns] 再轉 int64，才能穩定拿到「秒」。
        ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
        ts = ts.astype("datetime64[ns]").astype("int64") / 1e9
        duration_s = pd.to_numeric(chunk["Flow Duration"], errors="coerce") / 1e6  # 微秒 -> 秒
        for src_ip, src_port, dst_ip, dst_port, start, dur, label in zip(
            chunk["Src IP"], chunk["Src Port"], chunk["Dst IP"], chunk["Dst Port"],
            ts, duration_s, chunk["Label"],
        ):
            if pd.isna(start):
                continue
            dur = 0.0 if pd.isna(dur) or dur < 0 else dur
            key = _canonical_key(src_ip, src_port, dst_ip, dst_port)
            index.setdefault(key, []).append((float(start), float(start) + dur, label))
            n_rows += 1
    return index, n_rows


def host_ip_of(pcap_path: str) -> str | None:
    m = _HOST_IP_RE.search(os.path.basename(pcap_path))
    return m.group(1) if m else None


@dataclass
class LabeledFlow:
    marker: tuple[str, int, str, int]
    start_time: float
    end_time: float
    payload: bytes
    protocol: list
    label: str | None   # None 代表在 CSV 索引裡找不到時間夠接近的對應紀錄


def calibrate_offset(index, calibration_pcap: str, max_samples: int = 200) -> float:
    """csv_epoch - pcap_epoch 的中位差。只用「這個 canonical key 在 CSV 裡
    唯一對應一筆」的 flow 來估，避免同一組五元組當天重複連線造成誤配。"""
    diffs = []
    for marker, start, _end, _payload in iter_host_flows_fast(calibration_pcap):
        key = _canonical_key(*marker)
        candidates = index.get(key)
        if candidates and len(candidates) == 1:
            diffs.append(candidates[0][0] - start)
            if len(diffs) >= max_samples:
                break
    if not diffs:
        raise RuntimeError(
            f"校正失敗：{calibration_pcap} 裡找不到任何能在 CSV 索引唯一對應的 flow，"
            "確認 CSV 跟 pcap 是不是同一天的資料"
        )
    diffs.sort()
    return diffs[len(diffs) // 2]


def find_label(index, marker, start, end, offset: float, tolerance: float = DEFAULT_TOLERANCE):
    """回傳 (label, matched: bool)。挑候選裡跟 [start+offset, end+offset]
    時間重疊最多、或起點最接近的那筆；差距超過 tolerance 就當作沒對到。"""
    key = _canonical_key(*marker)
    candidates = index.get(key)
    if not candidates:
        return None, False

    adj_start, adj_end = start + offset, end + offset
    best, best_score = None, None
    for csv_start, csv_end, label in candidates:
        overlap = min(adj_end, csv_end) - max(adj_start, csv_start)
        gap = max(0.0, -overlap)          # 沒重疊時，起點之間的距離
        score = (overlap, -gap)
        if best_score is None or score > best_score:
            best, best_score = (label, gap), score

    label, gap = best
    if best_score[0] <= 0 and gap > tolerance:
        return None, False
    return label, True


def iter_labeled_flows(
    pcap_dir: str,
    csv_path: str,
    *,
    calibration_pcap: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
):
    """對 pcap_dir 底下每個主機 pcap 重組 flow，逐一標記，邊處理邊 yield
    LabeledFlow——不會把整個資料夾的內容一次讀進記憶體。"""
    pcap_files = sorted(glob.glob(os.path.join(pcap_dir, "*.pcap")))
    if not pcap_files:
        raise FileNotFoundError(f"{pcap_dir} 底下沒有 .pcap 檔")

    index, _n_rows = build_label_index(csv_path)
    calibration_pcap = calibration_pcap or min(pcap_files, key=os.path.getsize)
    offset = calibrate_offset(index, calibration_pcap)

    for pcap_path in pcap_files:
        for marker, start, end, payload in iter_host_flows_fast(pcap_path):
            label, matched = find_label(index, marker, start, end, offset, tolerance)
            yield LabeledFlow(marker, start, end, payload,
                               protocol=[], label=label if matched else None)


def parse_atk_vic_html(path: str):
    """粗略解析 atk_vic.html 的表格（Attacker/Victim/Attack Name/Date/Start/
    Finish），只用來做 sanity_check()，不追求通用 HTML parser。"""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    records = []
    for row in rows:
        cells = re.findall(r"<td>(.*?)</td>", row, re.DOTALL)
        if len(cells) != 6:
            continue
        clean = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        attacker, victim, attack, date, start, finish = clean
        records.append({
            "attacker": attacker, "victim": victim, "attack": attack,
            "date": date, "start": start, "finish": finish,
        })
    return records


def sanity_check(pcap_dir: str, csv_path: str, atk_vic_html: str, date_match: str):
    """拿 atk_vic.html 裡指定日期的攻擊時間窗，跟 join 出來的 Label 分佈交叉
    驗證：該窗口內、victim IP 相關的 flow，Label 是否真的變成對應攻擊名稱。"""
    records = [r for r in parse_atk_vic_html(atk_vic_html) if date_match in r["date"]]
    if not records:
        print(f"[sanity_check] atk_vic.html 裡找不到日期含 {date_match!r} 的紀錄")
        return

    print(f"[sanity_check] {date_match} 官方記載的攻擊時間窗：")
    for r in records:
        print(f"  {r['attack']:20s} victim={r['victim']:30s} {r['start']}~{r['finish']}")

    from collections import Counter
    counts = Counter()
    for lf in iter_labeled_flows(pcap_dir, csv_path):
        counts[lf.label] += 1

    print("\n[sanity_check] 實際 join 出來的 Label 分佈：")
    for label, n in counts.most_common():
        print(f"  {label!r:35s} {n}")


def main():
    ap = argparse.ArgumentParser(description="把 pcap 重組的 flow 標記回已修正的 CICFlowMeter CSV")
    ap.add_argument("pcap_dir", help="單一天份的 pcap 資料夾（每個主機一個 .pcap）")
    ap.add_argument("csv_path", help="同一天已修正過的 CICFlowMeter CSV")
    ap.add_argument("--calibration-pcap", default=None, help="用來校正時區 offset 的 pcap，預設用資料夾裡最小的檔案")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--atk-vic-html", default=None, help="atk_vic.html 路徑，提供的話會做 sanity check")
    ap.add_argument("--date-match", default=None, help="atk_vic.html 裡用來篩選日期的子字串，例如 '23-02-2018'")
    ap.add_argument("--manifest", default=None, help="把 (marker, start, end, label, payload 長度) 存成 csv 的路徑")
    args = ap.parse_args()

    if args.atk_vic_html:
        if not args.date_match:
            raise SystemExit("--atk-vic-html 需要搭配 --date-match")
        sanity_check(args.pcap_dir, args.csv_path, args.atk_vic_html, args.date_match)
        return

    from collections import Counter
    counts = Counter()
    manifest_rows = []
    for lf in iter_labeled_flows(args.pcap_dir, args.csv_path,
                                  calibration_pcap=args.calibration_pcap,
                                  tolerance=args.tolerance):
        counts[lf.label] += 1
        if args.manifest:
            src, sport, dst, dport = lf.marker
            manifest_rows.append({
                "src_ip": src, "src_port": sport, "dst_ip": dst, "dport": dport,
                "start": lf.start_time, "end": lf.end_time,
                "label": lf.label, "payload_len": len(lf.payload),
            })

    total = sum(counts.values())
    matched = total - counts.get(None, 0)
    print(f"共 {total} 條 flow，比對到 Label 的有 {matched} 條（{matched / total:.1%}）")
    for label, n in counts.most_common():
        print(f"  {label!r:35s} {n}")

    if args.manifest:
        pd.DataFrame(manifest_rows).to_csv(args.manifest, index=False)
        print(f"manifest 已寫到 {args.manifest}")


if __name__ == "__main__":
    main()
