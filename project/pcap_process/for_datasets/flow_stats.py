"""
跟 for_pcap.py 的 Flow / FlowPackager 平行的統計追蹤器：對每條 TCP flow
累積 CICFlowMeter 那組統計特徵（model/LightGBM 訓練用的 83 個欄位），而不是
像 for_pcap.py 那樣重組 byte stream。

兩者刻意分開、各自獨立維護狀態：FlowPackager 是給 textCNN 用的（要 byte
stream），這裡是給 LightGBM 用的（要統計量）。online.py 的 sniffer callback
對同一個封包會同時呼叫兩邊的 add()，互不干擾。

實作依據（拿 pcap/pcap2-23/ 的真實封包 + 對應的 Friday-23-02-2018.csv 逐欄
核對出來的，不是憑印象猜的）：
  * Flow Duration / IAT / Active / Idle 都是「微秒」；Active/Idle 的切段門檻
    是 5,000,000 微秒（5 秒），跟 CNS2022_Code README 記載的
    activityTimeout 一致。
  * "Total Length of Fwd/Bwd Packet"、"Packet Length *"、"Average Packet
    Size"、"Fwd/Bwd Segment Size Avg" 這些名字裡有「Packet／Size」的欄位，
    實際算的都是 TCP payload（segment）位元組數，不含 IP／TCP header。
  * "Fwd/Bwd Header Length" 只計 TCP header（dataofs*4），不含 IP header。
  * "Fwd Seg Size Min" 實測是 forward 方向 TCP header 長度的最小值（不是
    payload 大小的最小值）——這是 CICFlowMeter 系列工具一個常被誤解的欄位。
  * "Subflow Fwd/Bwd Packets/Bytes" 這個 fork 的算法其實有 bug：除數用的是
    「這條 flow 的封包總數 (fwd+bwd)」而不是真正的 subflow 數，所以
    Subflow *Packets 幾乎恆為 0（整數除法，分子必然小於分母）。這裡刻意複製
    同一個（有 bug 的）算法，這樣即時算出來的值才會跟訓練資料同分佈，不會
    多一個 train/serve skew 來源。
  * "Total TCP Flow Time" 實測恆等於 "Flow Duration"。
  * "Down/Up Ratio" = Total_Bwd_Packets / Total_Fwd_Packets（浮點數）。
  * "Average Packet Size" = 全部 payload 位元組數 / 全部封包數（雙向合併）。

  * Bulk（Fwd/Bwd Bytes/Packet/Bulk Rate Avg）：同方向、間隔 <= 1 秒、且都
    帶 payload 的連續 >= 4 個封包算一個 bulk；但「連續」是看整條 flow 的封包
    順序，不是只看單一方向自己的封包順序——另一個方向插進一個帶 payload 的
    封包會打斷目前這個方向的 run（不帶 payload 的純 ACK 不影響任何方向的
    run）。一開始沒考慮跨方向打斷，拿真實資料核對時發現一堆本來該是 0 的
    值被算成非 0，才抓到這個問題。
  * Active/Idle：只在「真的發生閒置轉換」（同一條 flow 連續兩個封包間隔超過
    5 秒）時才記錄一筆；一條從頭到尾都沒閒置超過 5 秒的 flow，Active/Idle
    四個統計量全部是 0，不會把整條 flow 的時長當成一段 active——這點一開始
    也做錯了（誤把收尾時還沒觸發門檻的那段也算進 active），拿真實資料核對
    時發現幾乎每條 flow 的 Active Mean 都被錯誤地等於 Flow Duration 才抓到。

以上這幾點都是先寫出第一版、拿 pcap/pcap2-23/ 的真實封包重算一遍、跟
Friday-23-02-2018.csv 逐欄位對答案，抓出差異後修正的，不是單靠看文件猜的。
"""

from __future__ import annotations

import time
from collections import deque
from threading import RLock

import numpy as np
from scapy.all import IP, TCP

ACTIVITY_TIMEOUT_US = 5_000_000   # 秒轉微秒；跟 CNS2022_Code README 的 activityTimeout 一致
BULK_GAP_US = 1_000_000           # 同一個 bulk 內，同方向連續封包間隔上限
BULK_MIN_PKTS = 4                 # 至少要幾個連續封包才算一個 bulk

_FLAG_BITS = {
    "FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08,
    "ACK": 0x10, "URG": 0x20, "ECE": 0x40, "CWR": 0x80,
}


def _stat4(xs):
    """回傳 (mean, std, max, min)；空 list 全部是 0.0。std 用樣本標準差
    （分母 n-1），跟 CICFlowMeter 的 DescriptiveStatistics 一致。"""
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(xs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std, float(arr.max()), float(arr.min())


class _Dir:
    """單一方向（fwd 或 bwd）的封包累積量。

    bulk 的「連續同方向」run 需要看兩個方向交錯的狀況（另一個方向插進一個帶
    payload 的封包，就會打斷這個方向的 run），所以 run 的判斷邏輯不在這裡，
    在 FlowStats 那層做；這裡只負責記錄 run 結束後的累計結果。
    """

    __slots__ = ("lengths", "header_lengths", "iats", "last_time",
                 "psh", "urg", "rst", "fin", "init_win", "last_win", "act_data_pkts",
                 "bulk_bytes_total", "bulk_pkts_total", "bulk_count", "bulk_duration_total")

    def __init__(self):
        self.lengths = []          # payload bytes，每個封包一筆
        self.header_lengths = []   # TCP header bytes，每個封包一筆
        self.iats = []             # 跟「同方向」上一個封包的時間差（微秒）
        self.last_time = None
        self.psh = self.urg = self.rst = self.fin = 0
        self.init_win = -1
        self.last_win = -1
        self.act_data_pkts = 0
        self.bulk_bytes_total = 0
        self.bulk_pkts_total = 0
        self.bulk_count = 0
        self.bulk_duration_total = 0

    def add(self, ts_us, length, header_len, flags, win):
        if self.last_time is not None:
            self.iats.append(ts_us - self.last_time)
        self.last_time = ts_us

        self.lengths.append(length)
        self.header_lengths.append(header_len)
        if flags & 0x08:
            self.psh += 1
        if flags & 0x20:
            self.urg += 1
        if flags & 0x04:
            self.rst += 1
        if flags & 0x01:
            self.fin += 1
        if self.init_win < 0:
            self.init_win = win
        self.last_win = win
        if length > 0:
            self.act_data_pkts += 1

    def record_bulk_run(self, total_bytes, n_pkts, duration_us):
        if n_pkts >= BULK_MIN_PKTS:
            self.bulk_bytes_total += total_bytes
            self.bulk_pkts_total += n_pkts
            self.bulk_count += 1
            self.bulk_duration_total += max(1, duration_us)

    def bulk_avg(self):
        if self.bulk_count == 0:
            return 0.0, 0.0, 0.0
        bytes_avg = self.bulk_bytes_total / self.bulk_count
        pkts_avg = self.bulk_pkts_total / self.bulk_count
        rate_avg = (self.bulk_bytes_total / (self.bulk_duration_total / 1e6)
                    if self.bulk_duration_total else 0.0)
        return bytes_avg, pkts_avg, rate_avg


class FlowStats:
    """一條雙向 TCP flow 的 CICFlowMeter 統計累積量。

    fwd_marker（第一個看到的封包的 (src, sport, dst, dport)）決定哪個方向是
    "forward"——跟 CICFlowMeter／pcap_label.py 用同一個慣例：誰先發封包誰是
    Src。之後收到的封包只要 (src,sport,dst,dport) 對得上 fwd_marker 就算
    fwd，否則算 bwd（同一條 flow 只會有這兩種可能）。
    """

    def __init__(self, fwd_marker):
        self.fwd_marker = fwd_marker
        self.fwd = _Dir()
        self.bwd = _Dir()
        self.start_time = None
        self.last_time = None
        self.all_times = []          # 不分方向，每個封包的時間戳，算 Flow IAT 用
        self.flag_counts = {k: 0 for k in _FLAG_BITS}
        self.active_durations = []
        self.idle_durations = []
        self._seg_start = None
        self._seg_last = None
        self.closed = False
        self.pending_close = False   # 看到 RST／雙向 FIN，但還沒真的收尾

        # bulk run 狀態（跨方向）：目前這個 run 屬於哪個方向、run 的起訖時間
        # /位元組數/封包數。另一個方向插進一個帶 payload 的封包會打斷目前的
        # run；不帶 payload 的封包（純 ACK）完全不影響 run，兩個方向都一樣。
        self._bulk_dir = None        # True=fwd, False=bwd, None=目前沒有進行中的 run
        self._bulk_start = None
        self._bulk_last = None
        self._bulk_bytes = 0
        self._bulk_pkts = 0

    def add(self, ts_us, is_fwd, length, header_len, flags, win):
        if self.start_time is None:
            self.start_time = ts_us
        self.last_time = ts_us
        self.all_times.append(ts_us)

        d = self.fwd if is_fwd else self.bwd
        d.add(ts_us, length, header_len, flags, win)

        for name, bit in _FLAG_BITS.items():
            if flags & bit:
                self.flag_counts[name] += 1

        if length > 0:
            self._bulk_update(is_fwd, ts_us, length)

        if self._seg_start is None:
            self._seg_start = ts_us
        elif ts_us - self._seg_last > ACTIVITY_TIMEOUT_US:
            self.active_durations.append(self._seg_last - self._seg_start)
            self.idle_durations.append(ts_us - self._seg_last)
            self._seg_start = ts_us
        self._seg_last = ts_us

    def _bulk_update(self, is_fwd, ts_us, length):
        same_dir = self._bulk_dir is not None and self._bulk_dir == is_fwd
        if same_dir and ts_us - self._bulk_last <= BULK_GAP_US:
            self._bulk_bytes += length
            self._bulk_pkts += 1
            self._bulk_last = ts_us
        else:
            self._bulk_flush()
            self._bulk_dir = is_fwd
            self._bulk_start = ts_us
            self._bulk_last = ts_us
            self._bulk_bytes = length
            self._bulk_pkts = 1

    def _bulk_flush(self):
        if self._bulk_dir is not None:
            d = self.fwd if self._bulk_dir else self.bwd
            d.record_bulk_run(self._bulk_bytes, self._bulk_pkts, self._bulk_last - self._bulk_start)
        self._bulk_dir = None
        self._bulk_bytes = 0
        self._bulk_pkts = 0

    def finalize(self):
        self._bulk_flush()
        # 注意：跟一般直覺不同，這裡「不」把最後一段還沒觸發 idle 門檻的
        # active 區段補進 active_durations——實測 CICFlowMeter fork 的
        # Active/Idle 只在真的發生「閒置轉換」時才記錄，一條從頭到尾都在
        # activityTimeout 內、從未閒置超過 5 秒的 flow，Active/Idle 全部是
        # 0，不是「把整條 flow 時長當一段 active」。
        self.closed = True

    def to_features(self) -> dict:
        src, sport, dst, dport = self.fwd_marker
        duration_us = (self.last_time - self.start_time) if self.last_time is not None else 0.0
        duration_s = duration_us / 1e6

        fwd_n, bwd_n = len(self.fwd.lengths), len(self.bwd.lengths)
        total_n = fwd_n + bwd_n
        fwd_bytes, bwd_bytes = sum(self.fwd.lengths), sum(self.bwd.lengths)
        total_bytes = fwd_bytes + bwd_bytes

        fwd_mean, fwd_std, fwd_max, fwd_min = _stat4(self.fwd.lengths)
        bwd_mean, bwd_std, bwd_max, bwd_min = _stat4(self.bwd.lengths)
        pkt_mean, pkt_std, pkt_max, pkt_min = _stat4(self.fwd.lengths + self.bwd.lengths)

        times = sorted(self.all_times)
        flow_iats = [b - a for a, b in zip(times, times[1:])]
        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _stat4(flow_iats)

        fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _stat4(self.fwd.iats)
        bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _stat4(self.bwd.iats)

        active_mean, active_std, active_max, active_min = _stat4(self.active_durations)
        idle_mean, idle_std, idle_max, idle_min = _stat4(self.idle_durations)

        fwd_bulk_bytes, fwd_bulk_pkts, fwd_bulk_rate = self.fwd.bulk_avg()
        bwd_bulk_bytes, bwd_bulk_pkts, bwd_bulk_rate = self.bwd.bulk_avg()

        return {
            "Dst Port": dport,
            "Protocol": 6,  # 只處理 TCP
            "Flow Duration": duration_us,
            "Total Fwd Packet": fwd_n,
            "Total Bwd packets": bwd_n,
            "Total Length of Fwd Packet": fwd_bytes,
            "Total Length of Bwd Packet": bwd_bytes,
            "Fwd Packet Length Max": fwd_max,
            "Fwd Packet Length Min": fwd_min,
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": fwd_std,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Min": bwd_min,
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": bwd_std,
            "Flow Bytes/s": (total_bytes / duration_s) if duration_s > 0 else 0.0,
            "Flow Packets/s": (total_n / duration_s) if duration_s > 0 else 0.0,
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            "Fwd IAT Total": sum(self.fwd.iats),
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Total": sum(self.bwd.iats),
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd PSH Flags": self.fwd.psh,
            "Bwd PSH Flags": self.bwd.psh,
            "Fwd URG Flags": self.fwd.urg,
            "Bwd URG Flags": self.bwd.urg,
            "Fwd RST Flags": self.fwd.rst,
            "Bwd RST Flags": self.bwd.rst,
            "Fwd Header Length": sum(self.fwd.header_lengths),
            "Bwd Header Length": sum(self.bwd.header_lengths),
            "Fwd Packets/s": (fwd_n / duration_s) if duration_s > 0 else 0.0,
            "Bwd Packets/s": (bwd_n / duration_s) if duration_s > 0 else 0.0,
            "Packet Length Min": pkt_min,
            "Packet Length Max": pkt_max,
            "Packet Length Mean": pkt_mean,
            "Packet Length Std": pkt_std,
            "Packet Length Variance": pkt_std ** 2,
            "FIN Flag Count": self.flag_counts["FIN"],
            "SYN Flag Count": self.flag_counts["SYN"],
            "RST Flag Count": self.flag_counts["RST"],
            "PSH Flag Count": self.flag_counts["PSH"],
            "ACK Flag Count": self.flag_counts["ACK"],
            "URG Flag Count": self.flag_counts["URG"],
            "CWR Flag Count": self.flag_counts["CWR"],
            "ECE Flag Count": self.flag_counts["ECE"],
            "Down/Up Ratio": (bwd_n / fwd_n) if fwd_n else 0.0,
            "Average Packet Size": (total_bytes / total_n) if total_n else 0.0,
            "Fwd Segment Size Avg": fwd_mean,
            "Bwd Segment Size Avg": bwd_mean,
            "Fwd Bytes/Bulk Avg": fwd_bulk_bytes,
            "Fwd Packet/Bulk Avg": fwd_bulk_pkts,
            "Fwd Bulk Rate Avg": fwd_bulk_rate,
            "Bwd Bytes/Bulk Avg": bwd_bulk_bytes,
            "Bwd Packet/Bulk Avg": bwd_bulk_pkts,
            "Bwd Bulk Rate Avg": bwd_bulk_rate,
            # 見檔案開頭說明：這個 fork 的 Subflow *Packets 實測恆為 0（bug），
            # 這裡刻意複製同一個算法（除以 fwd+bwd 總封包數，不是真正的
            # subflow 數），避免跟訓練資料分佈不一致
            "Subflow Fwd Packets": (fwd_n // total_n) if total_n else 0,
            "Subflow Fwd Bytes": (fwd_bytes // total_n) if total_n else 0,
            "Subflow Bwd Packets": (bwd_n // total_n) if total_n else 0,
            "Subflow Bwd Bytes": (bwd_bytes // total_n) if total_n else 0,
            "FWD Init Win Bytes": self.fwd.init_win,
            # 跟 FWD 不對稱：實測這個 fork 的 Bwd Init Win Bytes 對不上第一個
            # bwd 封包（SYN-ACK）的 window，反而對得上 bwd 方向「最後一個」
            # 封包的 window（TCP 接收視窗通常隨連線進行往上調，SYN-ACK 當下
            # 的值只是還沒調整完成的初始值）。用真實資料核對出來的，不是猜的
            "Bwd Init Win Bytes": self.bwd.last_win,
            "Fwd Act Data Pkts": self.fwd.act_data_pkts,
            "Fwd Seg Size Min": min(self.fwd.header_lengths) if self.fwd.header_lengths else 0,
            "Active Mean": active_mean,
            "Active Std": active_std,
            "Active Max": active_max,
            "Active Min": active_min,
            "Idle Mean": idle_mean,
            "Idle Std": idle_std,
            "Idle Max": idle_max,
            "Idle Min": idle_min,
            "ICMP Code": -1,   # 只處理 TCP flow，不會有 ICMP
            "ICMP Type": -1,
            "Total TCP Flow Time": duration_us,
        }


class FlowStatsTracker:
    """跟 online.py 的 LiveFlowPackager 平行：吃封包，維護每條 flow 的
    FlowStats，用真實時鐘的 idle timeout／FIN／RST 收尾。內建鎖，嗅探執行緒
    呼叫 add()、timer 執行緒呼叫 consume()，可以直接從兩個執行緒安全呼叫。
    """

    def __init__(self, idle_timeout: float = 15.0, close_grace: float = 2.0):
        self.idle_timeout = idle_timeout
        self.close_grace = close_grace  # 見 add() 的說明
        self._flows: dict[frozenset, FlowStats] = {}
        self._done: deque[FlowStats] = deque()
        self._lock = RLock()

    @staticmethod
    def _canonical_key(marker):
        src, sport, dst, dport = marker
        return frozenset({(src, sport), (dst, dport)})

    def add(self, pkt):
        tcp = pkt.getlayer(TCP)
        ip = pkt.getlayer(IP)
        if tcp is None or ip is None:
            return

        marker = (ip.src, tcp.sport, ip.dst, tcp.dport)
        key = self._canonical_key(marker)
        ts_us = float(pkt.time) * 1e6
        header_len = int(tcp.dataofs) * 4 if tcp.dataofs else 20
        # 不能直接用 len(bytes(tcp.payload))：短封包會被墊到 Ethernet 最小
        # frame 長度（64 bytes），scapy 有時會把那段墊料也算進 tcp.payload
        # 裡，導致極短的（例如純 ACK）封包被誤算出不存在的 payload。用
        # IP 表頭自己記錄的總長度（ip.len）往回推，才是封包實際攜帶的位元組
        # 數，不含物理層墊料。
        if ip.len:
            length = max(0, int(ip.len) - int(ip.ihl) * 4 - header_len)
        else:
            length = len(bytes(tcp.payload)) if tcp.payload else 0
        flags = int(tcp.flags)
        win = int(tcp.window)

        with self._lock:
            stats = self._flows.get(key)
            if stats is None:
                stats = self._flows[key] = FlowStats(marker)
            is_fwd = marker == stats.fwd_marker
            stats.add(ts_us, is_fwd, length, header_len, flags, win)

            if flags & _FLAG_BITS["RST"]:
                stats.pending_close = True
            elif (flags & _FLAG_BITS["FIN"]) and stats.fwd.fin > 0 and stats.bwd.fin > 0:
                # 跟 CICFlowMeter fork 的行為一致（README 提到的其中一個修
                # 正）：要雙向都送過 FIN 才算真正收尾，單邊 FIN 不算——單邊
                # FIN 常常只是半關閉（half-close），另一側可能還在傳資料
                stats.pending_close = True

            # 注意：這裡「不」馬上把 flow 從 self._flows 移除。RST／雙向 FIN
            # 之後常常還會有一個收尾用的封包（例如四向 FIN 交握最後一個單純
            # 的 ACK、或重傳的 RST），如果立刻 pop 掉，那個封包會被誤判成
            #「新的一條 flow」，導致同一條連線被切成兩段、封包數少算一個。
            # 改成只標記 pending_close，實際移除交給 consume() 用一個很短的
            # close_grace 寬限期處理——寬限期內沒有新封包才真的收尾。

    def _retire(self, key):
        stats = self._flows.pop(key, None)
        if stats is None:
            return
        stats.finalize()
        self._done.append(stats)

    def consume(self, now_us: float | None = None) -> list[tuple[tuple, dict]]:
        """收掉閒置 / 已收齊的 flow，回傳 [(fwd_marker, features), ...]。

        now_us 平常留 None，用真實時鐘；離線驗證/測試時可以傳進去用封包時間
        戳模擬，不用等真的時間流逝。
        """
        with self._lock:
            if now_us is None:
                now_us = time.time() * 1e6
            stale = []
            for k, s in self._flows.items():
                if s.last_time is None:
                    continue
                grace = self.close_grace if s.pending_close else self.idle_timeout
                if now_us - s.last_time > grace * 1e6:
                    stale.append(k)
            for key in stale:
                self._retire(key)

            out = []
            while self._done:
                stats = self._done.popleft()
                out.append((stats.fwd_marker, stats.to_features()))
            return out

    def close(self):
        with self._lock:
            for key in list(self._flows):
                self._retire(key)
