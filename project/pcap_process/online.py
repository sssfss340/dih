"""
即時（連線）版的 for_pcap。

差別在於資料來源與收尾時機：
  * for_pcap  ── 讀 pcap 檔，用「封包時間戳」判斷 flow 閒置，不開執行緒。
  * online    ── 用 scapy 直接嗅探網卡，用 threading.Timer 以「真實時鐘」
                 定期把閒置 / 已收齊的 flow 收尾，並把重組好的資料交給
                 下游 handler。

TCP stream 重組本身（Payload / Flow / FlowPackager / FlowOutput）直接沿用
for_pcap，不重寫。

CNN 跟 LightGBM 未來會跑在不同電腦上，所以不直接呼叫模型，而是把同一批封包
分別餵給兩條各自獨立的追蹤器，各自轉成自己要的輸入格式後用 HTTP JSON POST
送到各自的推論服務：
  * CNN      ── LiveFlowPackager 重組 byte stream，可以吃片段，逐步累積著送
    （見 _FlowByteBuffer）。
  * LightGBM ── flow_stats.FlowStatsTracker 累積 CICFlowMeter 那組統計特徵
    （跟 for_pcap.py 的 Flow 平行、互不干擾），整條 flow 收齊才送一次。

用法（需系統管理員 / root，Windows 需裝 Npcap）：
    python -m pcap_process.online -i Wi-Fi --cnn-url http://cnn-host:8800/predict
    python -m pcap_process.online -i eth0 --bpf "tcp port 80" \\
        --cnn-url http://cnn-host:8800/predict --lgbm-url http://lgbm-host:8801/predict
"""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from threading import Timer, RLock

from scapy.all import AsyncSniffer

try:  # 以套件方式執行： python -m pcap_process.online
    from .for_datasets.for_pcap import ProtocolParser, FlowPackager
    from .for_datasets.flow_stats import FlowStatsTracker
except ImportError:  # 直接執行： python pcap_process/online.py
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pcap_process.for_datasets.for_pcap import ProtocolParser, FlowPackager
    from pcap_process.for_datasets.flow_stats import FlowStatsTracker


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


CNN_MAX_BYTES = 1500  # 跟 model.textCNN.tokenizer.ByteTokenizer 的預設 max_len 對齊


class _FlowByteBuffer:
    """CNN 用：把 LiveFlowPackager 逐次吐出的「新增片段」累積成「目前為止的
    完整 payload」（截到 CNN_MAX_BYTES）。這樣 CNN 推論服務本身不用記任何
    flow 狀態，每通 HTTP request 都是獨立、無狀態的——之後要疊多台 CNN 推論
    服務分流也不用處理 sticky routing。"""

    def __init__(self, max_len: int = CNN_MAX_BYTES):
        self.max_len = max_len
        self._buffers: dict[tuple, bytearray] = {}

    def update(self, marker, data: bytes, offset: int) -> bytes:
        buf = self._buffers.setdefault(marker, bytearray())
        if offset > len(buf):
            return bytes(buf)          # 理論上不會發生（take_contiguous 保證連續），防禦性跳過
        if offset < len(buf):
            data = data[len(buf) - offset:]
        if len(buf) < self.max_len:
            buf.extend(data[: self.max_len - len(buf)])
        return bytes(buf)

    def discard(self, marker):
        self._buffers.pop(marker, None)


def send_json(url: str, payload: dict, timeout: float = 2.0):
    """把 payload 編碼成 JSON POST 給遠端推論服務。失敗只印警告、回傳 None，
    不能讓下游模型服務斷線/逾時拖垮封包擷取本身。"""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        print(f"[online] POST {url} 失敗: {exc!r}")
        return None


def make_cnn_handler(cnn_url: str):
    """回傳 LiveFlowPackager 用的 on_flow callback：累積片段、送 base64
    payload 給 CNN 推論服務。"""
    cnn_buffers = _FlowByteBuffer()

    def handle(marker, data, protocol, offset, closed):
        buf = cnn_buffers.update(marker, data, offset)
        if closed:
            cnn_buffers.discard(marker)
        if not buf:
            return
        src, sport, dst, dport = marker
        result = send_json(cnn_url, {
            "src": src, "sport": sport, "dst": dst, "dport": dport,
            "closed": closed,
            "protocol": [[name, count] for name, count in protocol],
            # bytes 不能直接塞進 JSON，用 base64 帶過去；tokenize（bytes ->
            # id 序列）留給推論服務自己做，這裡不重複實作一份 ByteTokenizer
            "payload_b64": base64.b64encode(buf).decode("ascii"),
        })
        if result is not None:
            print(f"[cnn]  {src}:{sport} -> {dst}:{dport}  {result}")

    return handle


def send_lgbm_features(lgbm_url: str, fwd_marker, features: dict):
    src, sport, dst, dport = fwd_marker
    result = send_json(lgbm_url, {"features": features})
    if result is not None:
        print(f"[lgbm] {src}:{sport} -> {dst}:{dport}  {result}")


def run(iface=None, bpf="tcp", interval=2.0, idle_timeout=15.0,
        cnn_url=None, lgbm_url=None):
    if not cnn_url and not lgbm_url:
        print("[online] 提醒：--cnn-url 和 --lgbm-url 都沒給，會嗅探封包但不會送去任何地方")

    packager = None
    if cnn_url:
        parser = ProtocolParser()
        packager = LiveFlowPackager(parser, on_flow=make_cnn_handler(cnn_url), idle_timeout=idle_timeout)

    stats_tracker = FlowStatsTracker(idle_timeout=idle_timeout) if lgbm_url else None

    def flush_lgbm():
        for fwd_marker, features in stats_tracker.consume():
            try:
                send_lgbm_features(lgbm_url, fwd_marker, features)
            except Exception as exc:
                print(f"[online] lgbm handler error on {fwd_marker}: {exc!r}")

    def sniff_cb(pkt):
        # CNN 跟 LightGBM 各自維護自己的 flow 狀態，同一個封包分別餵給兩邊，
        # 互不干擾（其中一邊沒開就跳過）
        if packager is not None:
            packager.add(pkt)
        if stats_tracker is not None:
            stats_tracker.add(pkt)

    def tick():
        if packager is not None:
            packager.consume()
        if stats_tracker is not None:
            flush_lgbm()

    ticker = RepeatingTimer(interval, tick)
    ticker.daemon = True
    ticker.start()

    sniffer = AsyncSniffer(iface=iface, filter=bpf, store=False, prn=sniff_cb)
    sniffer.start()
    print(f"[online] sniffing iface={iface or 'default'} filter={bpf!r} "
          f"(Ctrl+C 結束)  cnn={'on' if cnn_url else 'off'} lgbm={'on' if lgbm_url else 'off'}")

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
        if packager is not None:
            packager.close()       # 把還開著的 flow 全部收尾
            packager.consume()     # 最後一次把 output 清乾淨
        if stats_tracker is not None:
            stats_tracker.close()
            flush_lgbm()


def main():
    ap = argparse.ArgumentParser(description="即時封包重組")
    ap.add_argument("-i", "--iface", default=None,
                    help="要嗅探的網卡名稱（省略則用 scapy 預設）")
    ap.add_argument("--bpf", default="tcp",
                    help="BPF 過濾字串，預設 'tcp'")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="timer 掃描 / 輸出間隔（秒），預設 2")
    ap.add_argument("--idle", type=float, default=15.0,
                    help="flow 閒置多久（秒）視為結束，預設 15")
    ap.add_argument("--cnn-url", default=None,
                    help="textCNN 推論服務的網址（例如 http://host:8800/predict），省略則不送")
    ap.add_argument("--lgbm-url", default=None,
                    help="LightGBM 推論服務的網址（例如 http://host:8801/predict），省略則不送")
    args = ap.parse_args()
    run(iface=args.iface, bpf=args.bpf,
        interval=args.interval, idle_timeout=args.idle,
        cnn_url=args.cnn_url, lgbm_url=args.lgbm_url)


if __name__ == "__main__":
    main()
