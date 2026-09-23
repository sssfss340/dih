"""
把 TextCNNIDS 包成一個最小的 HTTP 推論服務，只用標準庫（不引入新依賴）。

搭配 pcap_process/online.py 的 --cnn-url 使用：未來 CNN／LightGBM 會跑在
不同電腦上，這支服務就是 CNN 那台機器上常駐的程序，online.py 只需要
知道它的網址，不需要在同一個 process 裡載入模型。

用法：
    python -m model.textCNN.serve                 # 預設監聽 0.0.0.0:8800
    python -m model.textCNN.serve --port 9000

request：  POST /predict
    body: {"payload_b64": "<base64 payload>", ...其他欄位 online.py 會附上但這裡用不到}
response： {"label": "BENIGN"}
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # 以套件方式執行： python -m model.textCNN.serve
    from .train import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, TextCNNIDS
except ImportError:  # 直接執行
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from model.textCNN.train import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, TextCNNIDS


def make_handler(clf: TextCNNIDS):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # 安靜一點，不用把每個 request 都印出來洗版

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                payload = base64.b64decode(body["payload_b64"])
                label = clf.predict_one(payload)
                status, result = 200, {"label": label}
            except Exception as exc:
                status, result = 400, {"error": str(exc)}

            resp = json.dumps(result).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="textCNN 推論 HTTP 服務")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--meta", default=DEFAULT_META_PATH)
    args = ap.parse_args()

    clf = TextCNNIDS.load(args.model, args.meta)
    print(f"[cnn-serve] 已載入模型：max_len={clf.max_len}、{len(clf.classes)} 類 {clf.classes}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(clf))
    print(f"[cnn-serve] 監聽 http://{args.host}:{args.port}/predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[cnn-serve] stopping...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
