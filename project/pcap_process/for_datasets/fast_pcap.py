"""
for_pcap.py / flow_stats.py 的高速版本，專門給離線批次處理超大量 pcap 用
（例如 pcap_label.py 要處理的 68GB+ 資料）。

**為什麼要另開一個檔案，而不是直接改 for_pcap.py／flow_stats.py**：那兩個
檔案的 Flow / FlowStats 核心邏輯（seq 重組、統計累積）本來就只吃基本型別
（int/bytes/float），完全不依賴 scapy——真正慢的是呼叫端用 scapy 的
`PcapReader` + `pkt.getlayer(TCP)` 這種「完整解析成 Packet 物件」的過程。
實測（見對話紀錄）：光是 scapy 解析就佔了處理時間的 ~95% 以上，我們自己的
flow 重組／統計邏輯幾乎不花時間。所以這裡只重寫「怎麼從 pcap 拿到欄位」這一
層，底下的 Flow / FlowOutput / Payload / FlowStats / _Dir 全部直接 import
原本的來用，不重複實作，也不會有兩份邏輯不同步的風險。

`online.py` 的即時嗅探路徑（scapy AsyncSniffer）不受影響、不需要改：那條路
本來就要用 scapy 收即時封包，這裡的優化只是針對離線讀大量 pcap 檔案。

做法：用 `scapy.utils.RawPcapReader` 只拿每個封包的原始 bytes（不觸發 scapy
的 Packet dissection），自己用 struct 從固定 offset 解析 Ethernet／IPv4／
TCP header 需要的幾個欄位。只支援 Ethernet 連結層 + IPv4（這個資料集全部
都是），遇到其他情況直接跳過該封包。
"""

from __future__ import annotations

import socket
import struct

from scapy.utils import RawPcapReader

try:
    from .for_pcap import FLUSH_EVERY, GC_INTERVAL, IDLE_TIMEOUT, Flow, FlowOutput, Payload
    from .flow_stats import FlowStats, _FLAG_BITS
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from pcap_process.for_datasets.for_pcap import FLUSH_EVERY, GC_INTERVAL, IDLE_TIMEOUT, Flow, FlowOutput, Payload
    from pcap_process.for_datasets.flow_stats import FlowStats, _FLAG_BITS

_ETHERTYPE_IPV4 = 0x0800
_PROTO_TCP = 6
_ETH_HDR_LEN = 14


def iter_tcp_packets(pcap_path: str):
    """逐封包 yield (ts, src_ip, sport, dst_ip, dport, seq, flags, window,
    tcp_header_len, payload_bytes)。跳過非 IPv4／非 TCP 的封包。

    payload 長度用 IP header 自己記的總長度往回推（ip_total_len - ip_header
    - tcp_header），不是直接信任 raw bytes 到結尾——短封包會被墊到 Ethernet
    最小 frame 長度，直接切到結尾會把墊料當成 payload（跟 flow_stats.py 修
    過的同一個問題，這裡從一開始就避開）。
    """
    with RawPcapReader(pcap_path) as reader:
        for raw, meta in reader:
            if len(raw) < _ETH_HDR_LEN + 20:
                continue
            if struct.unpack_from("!H", raw, 12)[0] != _ETHERTYPE_IPV4:
                continue  # 只處理 IPv4

            ip_start = _ETH_HDR_LEN
            ver_ihl = raw[ip_start]
            ihl = (ver_ihl & 0x0F) * 4
            if raw[ip_start + 9] != _PROTO_TCP:
                continue  # 只處理 TCP
            ip_total_len = struct.unpack_from("!H", raw, ip_start + 2)[0]
            tcp_start = ip_start + ihl
            if len(raw) < tcp_start + 20:
                continue

            src_ip = socket.inet_ntoa(raw[ip_start + 12: ip_start + 16])
            dst_ip = socket.inet_ntoa(raw[ip_start + 16: ip_start + 20])
            sport, dport = struct.unpack_from("!HH", raw, tcp_start)
            seq = struct.unpack_from("!I", raw, tcp_start + 4)[0]
            dataofs = (raw[tcp_start + 12] >> 4) * 4
            flags = raw[tcp_start + 13]
            window = struct.unpack_from("!H", raw, tcp_start + 14)[0]

            payload_len = max(0, ip_total_len - ihl - dataofs)
            payload_start = tcp_start + dataofs
            payload = raw[payload_start: payload_start + payload_len]

            ts = meta.sec + meta.usec / 1e6
            yield ts, src_ip, sport, dst_ip, dport, seq, flags, window, dataofs, payload


class FastFlowPackager:
    """for_pcap.py FlowPackager 的高速版：邏輯（含 idle-timeout 的 _gc()）
    完全比照原版（直接重用 Flow / FlowOutput / Payload），只是輸入不是 scapy
    Packet，是 iter_tcp_packets() 吐出來的已解析欄位。不做協定辨識
    （HTTP/DNS/...）——那本身也要 scapy 的 layer dissection，是額外的慢，而
    pcap_label.py 只需要 payload bytes，協定標籤用不到。
    """

    def __init__(self):
        self.flows: dict[tuple, Flow] = {}
        self.output = FlowOutput()
        self._now = 0.0
        self._last_gc = 0.0

    def add(self, ts, src_ip, sport, dst_ip, dport, seq, flags, window, header_len, payload):
        if ts > self._now:
            self._now = ts

        marker = (src_ip, sport, dst_ip, dport)
        flow = self.flows.get(marker)
        if flow is None:
            flow = self.flows[marker] = Flow(marker)

        flow.add_segment(seq, Payload("", bytes(payload)),
                          syn=bool(flags & 0x02), fin=bool(flags & 0x01), ts=ts)

        if flags & 0x04:                                      # RST：直接收尾
            self._retire(marker)
        else:
            if (flags & 0x08) or flow.pending >= FLUSH_EVERY:  # PSH / 累積夠多
                self._mark(marker)
            if (flags & 0x01) and flow.can_complete():
                self._retire(marker)

        if self._now - self._last_gc >= GC_INTERVAL:
            self._gc()

    def close(self):
        for marker in list(self.flows):
            self._retire(marker)

    def _gc(self):
        self._last_gc = self._now
        stale = [m for m, f in self.flows.items()
                 if (f.last_time is not None and self._now - f.last_time > IDLE_TIMEOUT)
                 or f.can_complete()]
        for marker in stale:
            self._retire(marker)

    def _mark(self, marker):
        flow = self.flows.get(marker)
        if flow is not None and flow.pending:
            self.output.push(flow)
            flow.pending = 0

    def _retire(self, marker):
        flow = self.flows.pop(marker, None)
        if flow is None:
            return
        flow.closed = True
        if flow.saw_data:
            self.output.push(flow)
        flow.pending = 0


class FastFlowStatsTracker:
    """flow_stats.py FlowStatsTracker 的高速版：一樣直接重用 FlowStats，只是
    省掉執行緒鎖（離線批次單執行緒處理一個檔案，不需要）跟即時 wall-clock
    idle 判斷（離線用封包時間戳，讀完整個檔案再 close() 收尾就好，不需要
    consume() 的 idle sweep）。
    """

    def __init__(self):
        self._flows: dict[tuple, FlowStats] = {}
        self._done: list[FlowStats] = []

    def add(self, ts, src_ip, sport, dst_ip, dport, seq, flags, window, header_len, payload):
        marker = (src_ip, sport, dst_ip, dport)
        key = frozenset({(src_ip, sport), (dst_ip, dport)})
        stats = self._flows.get(key)
        if stats is None:
            stats = self._flows[key] = FlowStats(marker)
        is_fwd = marker == stats.fwd_marker
        stats.add(ts, is_fwd, len(payload), header_len, flags, window)

        if flags & _FLAG_BITS["RST"]:
            self._retire(key)
        elif (flags & _FLAG_BITS["FIN"]) and stats.fwd.fin > 0 and stats.bwd.fin > 0:
            self._retire(key)

    def _retire(self, key):
        stats = self._flows.pop(key, None)
        if stats is None:
            return
        stats.finalize()
        self._done.append(stats)

    def close(self):
        for key in list(self._flows):
            self._retire(key)

    def drain(self):
        out, self._done = self._done, []
        return out


def iter_host_flows_fast(pcap_path: str):
    """跟 pcap_label.py 的 iter_host_flows 對應，回傳 (marker, start_time,
    end_time, payload_bytes)，一次處理一個主機的 pcap（讀完整個檔案才收
    尾，檔案之間不延續狀態，跟原本的假設一致）。"""
    packager = FastFlowPackager()
    for pkt in iter_tcp_packets(pcap_path):
        packager.add(*pkt)
    packager.close()

    while packager.output:
        flow = packager.output.popleft()
        got = flow.take_contiguous()
        tail = flow.take_remainder()
        if got and tail:
            got = (got[0] + tail[0], got[1])
        elif tail:
            got = tail
        if got is None or flow.start_time is None:
            continue
        payload, _offset = got
        yield flow.marker, flow.start_time, flow.last_time, payload.content


def host_flow_stats_fast(pcap_path: str):
    """跟 flow_stats.py 對應的離線批次版：回傳這個主機 pcap 裡所有 flow 的
    (fwd_marker, features dict) list。"""
    tracker = FastFlowStatsTracker()
    for pkt in iter_tcp_packets(pcap_path):
        tracker.add(*pkt)
    tracker.close()
    return [(s.fwd_marker, s.to_features()) for s in tracker.drain()]
