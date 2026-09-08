from scapy.layers.http import HTTPRequest, HTTPResponse
from scapy.all import DNS, IP, TCP, PcapReader, Raw, Packet
import os
import re
from collections import defaultdict
from threading import Timer
from queue import Queue

if os.name == "nt":
    PS = "\\" # Path Separator
    LB = "\r\n" # Line Break
else:
    PS = "/"
    LB = "\n"

dir_path = f"{os.path.dirname(os.path.abspath(__file__))}{PS}pcap 2-23{PS}pcap"
# 副檔名為 .pcap 或純數字 (.1, .25 ...) 才處理
pattern = re.compile(r'^\.(?:\d+|pcap)$')
streams = defaultdict(list)

class RepeatingTimer(Timer):
    def run(self):
        self.finished.wait(self.interval)
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

class Payload:
    def __init__(self,protocol,content=b""):
        if not isinstance(content,bytes):
            raise TypeError("contect must be bytes")
        self.content=content
        if isinstance(protocol,str):
            self.protocol=[(protocol,1)]
        elif isinstance(protocol,list):
            self.protocol=protocol
        else:
            TypeError("protocol must be str or list")

    def __add__(self,other):
        if isinstance(other,Payload):
            if self.protocol == [("",1)]:
                return NotImplemented
            if self.protocol[-1][0] == other.protocol[0][0]:
                merge_protocol=self.protocol[:-1]+[(
                    self.protocol[-1][0],
                    self.protocol[-1][1]+other.protocol[0][1]
                    )]+other.protocol[1:]
            else:
                merge_protocol=self.protocol+other.protocol
            return Payload(merge_protocol,self.content+other.content)
        raise TypeError("addend must be class \'Payload\'")

    def __radd__(self, other):
        if other == "" or other.protocol == "":
            return self
        raise SyntaxError

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

parser=ProtocolParser()
for name in os.listdir(dir_path):
    base, ext = os.path.splitext(name)
    if not pattern.match(ext):
        continue
    packet=dir_path+PS+name

    for pkt in PcapReader(packet):
        payload = parser(pkt)
        if payload.content == b"":
            continue

        tcp = pkt.getlayer(TCP)
        if tcp is not None:
            ip = pkt.getlayer(IP)
            if ip is None:
                print(payload.content[:300])
                continue
            key = (ip.src, tcp.sport, ip.dst, tcp.dport)
            streams[key].append((tcp.seq, payload))
            continue

        print(payload.content.decode('utf-8', errors="replace")[:300])

    for key, segments in streams.items():
        # 按 TCP sequence number 排序後一次合併，避免 data += payload 的 O(n^2) 複製
        segments.sort(key=lambda x: x[0])
        data = b"".join(p.content for _, p in segments)

        print("=" * 80)
        print(key)
        print(data[:2000])

    print(base,"Fin")
    streams = defaultdict(list)
