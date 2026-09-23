from scapy.layers.http import HTTPRequest
from scapy.all import DNS, IP, TCP, PcapReader, Raw, Packet
import os
import re
from collections import deque

if os.name == "nt":
    PS = "\\" # Path Separator
    LB = "\r\n" # Line Break
else:
    PS = "/"
    LB = "\n"

dir_path = f"{os.path.dirname(os.path.abspath(__file__))}{PS}pcap 2-23{PS}pcap"
# 副檔名為 .pcap 或純數字 (.1, .25 ...) 才處理
pattern = re.compile(r'^\.(?:\d+|pcap)$')

# ---------------------------------------------------------------------------
# TCP stream 重組模型
#
# tcp.seq 是「位元組位移」，不是封包序號，起點是連線建立時隨機選的 ISN。
# 這裡對每個方向 (src, sport, dst, dport) 各開一個 Flow：
#   * base      = 相對位移的基準 seq（看得到 SYN 時 = ISN+1，否則 = 第一個看到的 seq）
#   * next_rel  = 下一個「尚未吐出」的連續位移
#   * buffered  = 亂序、還接不上的片段： rel_offset -> Payload
# 收到 PSH 或累積 FLUSH_EVERY 個有效 segment，就把該 Flow 排進 output。
# output.popleft() 回傳的是「Flow 物件本身」——也就是該 key 目前最新的 Flow，
# 就地更新，含之後才收到的 segment 與 closed 標記；同一個 Flow 在被取走前
# 重複排隊只會出現一次。連線收尾 (FIN 收齊 / RST / 閒置逾時 / 檔案讀完) 時，
# Flow 會被標記 closed 並從 self.flows 移除，但仍留在 output 佇列裡等人取走。
# 閒置逾時用封包時間戳判斷，不開執行緒。
# ---------------------------------------------------------------------------

SEQ_MOD      = 1 << 32
IDLE_TIMEOUT = 10.0        # 封包時間（秒）內沒有新 segment 就視為閒置
FLUSH_EVERY  = 5           # 累積幾個有效 segment 就把 Flow 排進 output
GC_INTERVAL  = 10.0        # 每隔多久（封包時間，秒）掃一次閒置 flow
MAX_GAP      = 4 << 20     # 相對 next_rel 超過這個距離的片段直接丟棄（視為 wrap / 擷取遺漏）


class FlowOutput:
    """FlowPackager 的輸出佇列。

    push() 進來的是 Flow 物件；同一個 Flow 在被 popleft() 取走前重複 push
    只會排一次。popleft() 回傳的永遠是該 Flow 目前的狀態（就地更新，含之後
    才 append 的 segment 與 closed 標記），不是當下的快照。
    """

    def __init__(self):
        self._queue   = deque()   # Flow（正常情況下每個 Flow 至多一項）
        self._pending = set()     # 目前排在佇列裡的 id(flow)

    def push(self, flow):
        fid = id(flow)
        if fid in self._pending:
            return
        self._pending.add(fid)
        self._queue.append(flow)

    def __len__(self):
        return len(self._pending)

    def __bool__(self):
        return bool(self._pending)

    def popleft(self):
        while self._queue:
            flow = self._queue.popleft()
            fid = id(flow)
            if fid in self._pending:
                self._pending.discard(fid)
                return flow
        raise IndexError("pop from empty FlowOutput")

    def clear(self):
        self._queue.clear()
        self._pending.clear()


class Payload:
    """一段位元組資料，附帶 run-length 形式的協定標註 [(name, count), ...]。"""

    def __init__(self, protocol, content=b""):
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content must be bytes")
        self.content = bytes(content)
        if isinstance(protocol, str):
            self.protocol = [(protocol, 1)] if protocol else []
        elif isinstance(protocol, list):
            self.protocol = list(protocol)
        else:
            raise TypeError("protocol must be str or list")

    def __add__(self, other):
        if not isinstance(other, Payload):
            return NotImplemented
        if not self.protocol:
            merged = list(other.protocol)
        elif not other.protocol:
            merged = list(self.protocol)
        elif self.protocol[-1][0] == other.protocol[0][0]:
            merged = (self.protocol[:-1]
                      + [(self.protocol[-1][0],
                          self.protocol[-1][1] + other.protocol[0][1])]
                      + other.protocol[1:])
        else:
            merged = self.protocol + other.protocol
        return Payload(merged, self.content + other.content)

    def __repr__(self):
        return f"Payload({self.protocol!r}, {self.content[:32]!r}...)"


class ProtocolParser:
    # 用一般 dict，避免 defaultdict 讀取時把未知 port 寫進來造成記憶體膨脹
    TCP_PACKET = {
        21 : "FTP",
        25 : "SMTP",
        587: "SMTP",
        110: "POP3",
        143: "IMAP",
    }

    PARSABLE = {"DNS", "HTTP", "FTP", "SMTP", "POP3", "IMAP"}

    def parse_DNS(self, dns):
        # 取得 DNS 查詢的域名與型態
        if dns.qr == 0 and dns.qd:
            try:
                qname = dns.qd.qname
                return Payload("DNS req", qname)
            except AttributeError:
                print("[Warning] DNS Question 沒有 qname")
                print(dns.show(dump=True))
                return Payload("DNS req")

        elif dns.qr == 1 and dns.an:
            ips = []
            for i in range(dns.ancount):
                answer = dns.an[i]
                if answer.type in (1, 28):   # A / AAAA
                    ips.append(str(answer.rdata))
            if ips:
                return Payload("DNS res", ", ".join(ips).encode())
        return Payload("DNS")

    def identify(self, dns, http_req, tcp):
        if dns is not None:
            return "DNS"
        if http_req is not None:
            return "HTTP"
        if tcp is not None:
            # 先看來源埠再看目的埠
            return self.TCP_PACKET.get(tcp.sport) or self.TCP_PACKET.get(tcp.dport, "")
        return ""

    def __call__(self, pkt):
        if not isinstance(pkt, Packet):
            raise TypeError("pkt must be class \'Packet\'")

        # 只抓一次 layer 參考後重用，避免 scapy 每次 pkt[X] 重走 layer chain
        dns      = pkt.getlayer(DNS)
        http_req = pkt.getlayer(HTTPRequest)
        tcp      = pkt.getlayer(TCP)
        raw      = pkt.getlayer(Raw)

        protocol = self.identify(dns, http_req, tcp)
        if protocol == "":
            return Payload("")
        if protocol == "DNS":
            return self.parse_DNS(dns)
        # HTTP / FTP / SMTP / POP3 / IMAP 都只是包 Raw payload
        return Payload(protocol, bytes(raw.load) if raw is not None else b"")


class Flow:
    """單一方向的 TCP byte-stream 重組緩衝區。"""

    __slots__ = ("marker", "base", "next_rel", "buffered", "pending",
                 "start_time", "last_time", "fin_rel", "closed", "saw_data")

    def __init__(self, marker):
        self.marker     = marker
        self.base       = None    # 相對位移的基準 seq
        self.next_rel   = 0       # 下一個尚未吐出的連續位移
        self.buffered   = {}      # rel_offset -> Payload（亂序、還接不上的片段）
        self.pending    = 0       # 距離上次排進 output 後新增的有效 segment 數
        self.start_time = None    # 第一個 segment 的封包時間戳（給離線標記比對用）
        self.last_time  = None    # 最後一個 segment 的封包時間戳
        self.fin_rel    = None    # FIN 對應的相對位移（= stream 結尾）
        self.closed     = False   # 連線已收尾，不會再有新 segment
        self.saw_data   = False   # 這條 flow 是否曾帶過 payload

    def _rel(self, seq):
        rel = seq - self.base
        if rel < 0:              # 單次 32-bit wraparound
            rel += SEQ_MOD
        return rel

    def add_segment(self, seq, payload, syn, fin, ts):
        if self.start_time is None:
            self.start_time = ts
        self.last_time = ts
        if self.base is None:
            # SYN 佔一個序號，資料從 seq+1 開始；沒 SYN 就把第一個 seq 當 byte 0
            self.base = (seq + 1) % SEQ_MOD if syn else seq

        rel  = self._rel(seq)
        data = payload.content

        if fin:
            self.fin_rel = rel + len(data)

        if not data:
            return
        self.saw_data = True

        end = rel + len(data)
        if end <= self.next_rel:
            return                          # 整段都是已吐出的資料（純重傳）
        if rel < self.next_rel:             # 部分重疊，切掉前面已吐出的
            cut = self.next_rel - rel
            payload = Payload(payload.protocol, data[cut:])
            rel = self.next_rel
        if rel > self.next_rel + MAX_GAP:
            return                          # 距離太遠，視為 wrap / 擷取遺漏，丟棄

        old = self.buffered.get(rel)
        if old is not None and len(old.content) >= len(payload.content):
            return                          # 已經有相同或更長的片段
        self.buffered[rel] = payload
        self.pending += 1

    def take_contiguous(self):
        """取出從 next_rel 開始接得起來的連續資料，回傳 (Payload, start_offset) 或 None。"""
        start = self.next_rel
        parts = []
        while True:
            seg = self.buffered.pop(self.next_rel, None)
            if seg is None:
                # 找一個「起點在前、跨過 next_rel」的重疊片段，切掉前半來補洞
                hit = None
                for off, buf in self.buffered.items():
                    if off < self.next_rel < off + len(buf.content):
                        hit = off
                        break
                if hit is None:
                    break
                seg = self.buffered.pop(hit)
                seg = Payload(seg.protocol, seg.content[self.next_rel - hit:])
            parts.append(seg)
            self.next_rel += len(seg.content)
        if not parts:
            return None
        merged = parts[0]
        for p in parts[1:]:
            merged = merged + p
        return merged, start

    def take_remainder(self):
        """把 buffer 裡剩下的亂序片段（中間可能有洞）依序併起來吐掉。"""
        if not self.buffered:
            return None
        offs   = sorted(self.buffered)
        start  = offs[0]
        merged = self.buffered.pop(start)
        for off in offs[1:]:
            merged = merged + self.buffered.pop(off)
        return merged, start

    def is_complete(self):
        return self.fin_rel is not None and self.next_rel >= self.fin_rel

    def can_complete(self):
        """FIN 已收到，且 buffer 裡的資料（不實際消費）連續覆蓋到 fin_rel。"""
        if self.fin_rel is None:
            return False
        pos = self.next_rel
        if pos >= self.fin_rel:
            return True
        for off in sorted(o for o in self.buffered if o >= pos):
            if off > pos:
                return False               # 中間有洞
            pos = max(pos, off + len(self.buffered[off].content))
            if pos >= self.fin_rel:
                return True
        return False


class FlowPackager:
    """吃封包，維護每個方向的 Flow，把有更新的 Flow 排進 output。

    self.flows[marker] 是該方向目前「活著」的 Flow。output.popleft() 取到的
    就是這個物件（或它剛被收尾、移出 self.flows 的最終狀態）。
    """

    def __init__(self, parser):
        self.parser   = parser
        self.flows    = {}              # marker -> 活著的 Flow
        self.output   = FlowOutput()
        self._now     = 0.0             # 目前看過最大的封包時間戳
        self._last_gc = 0.0

    def add(self, pkt):
        tcp = pkt.getlayer(TCP)
        ip  = pkt.getlayer(IP)
        if tcp is None or ip is None:
            return

        ts = float(pkt.time)
        if ts > self._now:
            self._now = ts

        marker = (ip.src, tcp.sport, ip.dst, tcp.dport)
        flow = self.flows.get(marker)
        if flow is None:
            flow = self.flows[marker] = Flow(marker)

        flags = int(tcp.flags)
        flow.add_segment(tcp.seq, self._payload(pkt, ip, tcp),
                         syn=bool(flags & 0x02),
                         fin=bool(flags & 0x01),
                         ts=ts)

        if flags & 0x04:                                     # RST：直接收尾
            self._retire(marker)
        else:
            if (flags & 0x08) or flow.pending >= FLUSH_EVERY:  # PSH / 累積夠多
                self._mark(marker)
            # 只在這個封包帶 FIN 時檢查收齊；「FIN 先到、破洞晚補」的情況交給 _gc
            if (flags & 0x01) and flow.can_complete():
                self._retire(marker)

        if self._now - self._last_gc >= GC_INTERVAL:
            self._gc()

    def close(self):
        """輸入讀完時呼叫：把還開著的 flow 全部收尾並排進 output。"""
        for marker in list(self.flows):
            self._retire(marker)

    # -- internal ----------------------------------------------------------

    def _payload(self, pkt, ip, tcp):
        dns = pkt.getlayer(DNS)
        if dns is not None:
            return self.parser.parse_DNS(dns)
        # 重組的是原始 TCP byte stream，協定只是附加標註
        label = self.parser.identify(None, pkt.getlayer(HTTPRequest), tcp)
        # 不能用 bytes(tcp.payload)：實測發現兩個各自獨立的問題（拿真實的
        # Friday-23-02-2018.csv 逐欄核對出來的）——
        #   1. 這批資料是從 AWS 的 NIC 擷取的，某些封包（尤其伺服器送出的大
        #      量資料）scapy 解析出來的 payload 長度比封包自己 IP header 記
        #      錄的 ip.len 大很多（很可能是 NIC/驅動層 GRO 之類的效果）；
        #   2. 更嚴重的是內容本身：scapy 辨識出 HTTP response 且是 gzip
        #      內容時，dissect 出來的 payload 有時是「已解壓縮」的版本，不
        #      是線路上真正傳輸的 bytes（等於偷偷把資料內容換掉了）。
        # 兩個問題的共同解法是別用 scapy dissect 後重建的 .payload，改用
        # pkt.original（scapy 保留、完全沒被 dissect 動過的原始 frame
        # bytes），自己照 ip.len 算出正確的起訖位置去切。
        ip_hdr_len = int(ip.ihl) * 4 if ip.ihl else 20
        tcp_hdr_len = int(tcp.dataofs) * 4 if tcp.dataofs else 20
        if ip.len and pkt.original:
            # eth_hdr_len 寫死 14：這批資料全部是標準 Ethernet 沒有 VLAN
            # tag，用 bytes(pkt)/bytes(ip) 反推長度會重新踩到上面說的 rebuild
            # 問題（bytes(pkt) 一樣會把 HTTP 那層重新序列化一次），所以不能
            # 用那個方式算，只能認資料集本身的框架格式。
            eth_hdr_len = 14
            start = eth_hdr_len + ip_hdr_len + tcp_hdr_len
            payload_len = max(0, int(ip.len) - ip_hdr_len - tcp_hdr_len)
            raw = bytes(pkt.original[start: start + payload_len])
        else:
            raw = bytes(tcp.payload) if tcp.payload else b""
        return Payload(label, raw)

    def _mark(self, marker):
        """該 flow 有新資料 -> 排進 output（已在佇列裡就不重複排）。"""
        flow = self.flows.get(marker)
        if flow is not None and flow.pending:
            self.output.push(flow)
            flow.pending = 0

    def _retire(self, marker):
        """收尾：標記 closed、移出 self.flows，仍留在 output 等人取走。"""
        flow = self.flows.pop(marker, None)
        if flow is None:
            return
        flow.closed = True
        if flow.saw_data:
            self.output.push(flow)         # push 對已排隊的 flow 是 no-op
        flow.pending = 0

    def _gc(self):
        self._last_gc = self._now
        stale = [m for m, f in self.flows.items()
                 if (f.last_time is not None and self._now - f.last_time > IDLE_TIMEOUT)
                 or f.can_complete()]
        for marker in stale:
            self._retire(marker)


def _pcap_sort_key(name):
    # 讓 foo.2 排在 foo.10 前面，而不是字典序
    base, ext = os.path.splitext(name)
    num = ext[1:]
    return (base, int(num)) if num.isdigit() else (base, -1)


def _drain(packager):
    while packager.output:
        flow = packager.output.popleft()          # 該 marker 目前最新的 Flow
        got = flow.take_contiguous()              # 自上次以來新接起來的連續資料
        if flow.closed:
            tail = flow.take_remainder()          # 中間有洞的殘餘也一起吐
            if got and tail:
                got = (got[0] + tail[0], got[1])
            elif tail:
                got = tail
        if got is None:
            continue
        payload, offset = got
        print("=" * 80)
        print(flow.marker, "offset", offset,
              "closed" if flow.closed else "", payload.protocol)
        print(payload.content[:2000])


def main():
    parser   = ProtocolParser()
    packager = FlowPackager(parser)

    names = sorted((n for n in os.listdir(dir_path)
                    if pattern.match(os.path.splitext(n)[1])),
                   key=_pcap_sort_key)

    for name in names:
        path = dir_path + PS + name
        with PcapReader(path) as reader:
            for pkt in reader:
                if pkt.haslayer(TCP) and pkt.haslayer(IP):
                    packager.add(pkt)
                elif pkt.haslayer(DNS):
                    pl = parser.parse_DNS(pkt.getlayer(DNS))
                    if pl.content:
                        print(pl.protocol, pl.content.decode("utf-8", "replace")[:300])
        _drain(packager)          # flow 跨檔案保留，只把已產生的資料印出來
        print(name, "Fin")

    packager.close()
    _drain(packager)


if __name__ == "__main__":
    main()
