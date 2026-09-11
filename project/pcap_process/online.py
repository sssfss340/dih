"""
即時（連線）版的 for_pcap。

差別在於資料來源與收尾時機：
  * for_pcap  ── 讀 pcap 檔，用「封包時間戳」判斷 flow 閒置，不開執行緒。
  * online    ── 用 scapy 直接嗅探網卡，用 threading.Timer 以「真實時鐘」
                 定期把閒置 / 已收齊的 flow 收尾，並把重組好的資料交給
                 下游 handler（預設印出；實務上換成寫 redis / 丟 lightGBM /
                 textCNN）。

TCP stream 重組本身（Payload / Flow / FlowPackager / FlowOutput）直接沿用
for_pcap，不重寫。

用法（需系統管理員 / root，Windows 需裝 Npcap）：
    python -m pcap_process.online -i Wi-Fi
    python -m pcap_process.online -i eth0 --bpf "tcp port 80" --interval 1 --idle 20
"""

import argparse
import time
from threading import Timer, RLock

from scapy.all import AsyncSniffer

try:  # 以套件方式執行： python -m pcap_process.online
    from .for_datasets.for_pcap import ProtocolParser, FlowPackager
except ImportError:  # 直接執行： python pcap_process/online.py
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pcap_process.for_datasets.for_pcap import ProtocolParser, FlowPackager


class RepeatingTimer(Timer):
    """每隔 interval 秒重複呼叫 function，直到 cancel()。"""

    def run(self):
        # finished.wait 回傳 True 代表 event 被 set（也就是被 cancel）
        while not self.finished.wait(self.interval):
            try:
                self.function(*self.args, **self.kwargs)
            except Exception as exc:                      # 別讓例外弄死 timer 執行緒
                print(f"[online] timer tick error: {exc!r}")


class LiveFlowPackager(FlowPackager):
    """FlowPackager + 執行緒鎖 + 真實時鐘的閒置逾時。

    嗅探執行緒呼叫 add()，timer 執行緒呼叫 consume()，兩者都經由同一把 RLock
    序列化。handler 只在鎖外執行，避免下游（redis / 模型推論）拖慢收包。
    """

    def __init__(self, parser, on_flow, idle_timeout=15.0):
        super().__init__(parser)
        self.on_flow      = on_flow
        self.idle_timeout = idle_timeout
        self._lock        = RLock()

    # -- 嗅探執行緒 ---------------------------------------------------------

    def add(self, pkt):
        with self._lock:
            super().add(pkt)

    # -- timer 執行緒 -----------------------------------------------------

    def consume(self):
        """收掉閒置 / 已收齊的 flow，把 output 佇列裡的資料抽出來交給 handler。"""
        batch = []
        with self._lock:
            # 即時擷取時 pkt.time ≈ 牆上時鐘，這裡直接用 time.time() 當基準，
            # 這樣就算完全沒有新封包進來，閒置的 flow 一樣會被收掉。
            self._now = time.time()
            self._gc()                       # 依 idle_timeout 收尾
            while self.output:
                flow = self.output.popleft()
                got = flow.take_contiguous()
                if flow.closed:
                    tail = flow.take_remainder()
                    if got and tail:
                        got = (got[0] + tail[0], got[1])
                    elif tail:
                        got = tail
                if got is None:
                    continue
                payload, offset = got
                batch.append((flow.marker, payload.content,
                              list(payload.protocol), offset, flow.closed))

        for marker, data, protocol, offset, closed in batch:
            try:
                self.on_flow(marker, data, protocol, offset, closed)
            except Exception as exc:
                print(f"[online] handler error on {marker}: {exc!r}")

    def close(self):
        with self._lock:
            super().close()

    # -- override：改用 self.idle_timeout，而不是 for_pcap 的全域常數 --------

    def _gc(self):
        self._last_gc = self._now
        stale = [m for m, f in self.flows.items()
                 if (f.last_time is not None
                     and self._now - f.last_time > self.idle_timeout)
                 or f.can_complete()]
        for marker in stale:
            self._retire(marker)


def handle_flow(marker, data, protocol, offset, closed):
    """下游接口。把這裡換成真正的處理即可，簽章不變。

    例：丟進 redis stream 讓推論服務去消費 ──
        import redis, json
        r = redis.Redis()
        r.xadd("flows", {
            "src": marker[0], "sport": marker[1],
            "dst": marker[2], "dport": marker[3],
            "offset": offset, "closed": int(closed),
            "protocol": json.dumps(protocol),
            "data": data,                       # bytes，redis 可直接存
        })
    或直接 lightGBM / textCNN：
        feats = extract_features(marker, data, protocol)
        score = model.predict([feats])[0]
    """
    src, sport, dst, dport = marker
    tag = "closed" if closed else "part"
    proto = "".join(f"{n}x{c}" for n, c in protocol) or "-"
    print("=" * 80)
    print(f"[{tag}] {src}:{sport} -> {dst}:{dport}  @{offset}  {proto}  {len(data)}B")
    print(data[:500])


def run(iface=None, bpf="tcp", interval=2.0, idle_timeout=15.0, on_flow=handle_flow):
    parser   = ProtocolParser()
    packager = LiveFlowPackager(parser, on_flow=on_flow, idle_timeout=idle_timeout)

    ticker = RepeatingTimer(interval, packager.consume)
    ticker.daemon = True
    ticker.start()

    sniffer = AsyncSniffer(iface=iface, filter=bpf, store=False, prn=packager.add)
    sniffer.start()
    print(f"[online] sniffing iface={iface or 'default'} filter={bpf!r} "
          f"(Ctrl+C 結束)")

    try:
        while sniffer.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[online] stopping...")
    finally:
        try:
            sniffer.stop()
        except Exception:
            pass
        ticker.cancel()
        packager.close()       # 把還開著的 flow 全部收尾
        packager.consume()     # 最後一次把 output 清乾淨


def main():
    ap = argparse.ArgumentParser(description="即時封包重組（for_pcap 的連線版）")
    ap.add_argument("-i", "--iface", default=None,
                    help="要嗅探的網卡名稱（省略則用 scapy 預設）")
    ap.add_argument("--bpf", default="tcp",
                    help="BPF 過濾字串，預設 'tcp'")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="timer 掃描 / 輸出間隔（秒），預設 2")
    ap.add_argument("--idle", type=float, default=15.0,
                    help="flow 閒置多久（秒）視為結束，預設 15")
    args = ap.parse_args()
    run(iface=args.iface, bpf=args.bpf,
        interval=args.interval, idle_timeout=args.idle)


if __name__ == "__main__":
    main()
