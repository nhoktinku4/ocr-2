"""Chạy API OCR GPKD (api.py), hoặc gửi file lên API đang chạy.

    python run_api.py                         # bật server ở 0.0.0.0:8002
    python run_api.py --port 9000 --reload    # đổi cổng, tự nạp lại khi sửa code (dev)
    python run_api.py a.pdf b.jpg             # gửi file lên API, in JSON kết quả
    python run_api.py a.pdf --url http://may-chu:8002
Biến môi trường API_HOST / API_PORT / API_URL cũng dùng được.
"""

import argparse
import json
import os
from pathlib import Path
import sys


def serve(args):
    import uvicorn
    # 1 worker: engine OCR/heron nạp một lần; request song song chạy trong threadpool.
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


def send(args):
    import httpx
    url = (args.url or os.getenv("API_URL") or f"http://127.0.0.1:{args.port}").rstrip("/") + "/ocr"
    params = {"include_text": args.text}
    # API trên chính máy này: gọi thẳng, không qua HTTP(S)_PROXY. Địa chỉ khác vẫn đi theo proxy của máy.
    local = httpx.URL(url).host in {"localhost", "127.0.0.1", "::1"}
    client = httpx.Client(trust_env=not local, timeout=args.timeout)
    failed = 0
    for path in args.files:
        path = Path(path).expanduser()
        try:
            with path.open("rb") as f:
                r = client.post(url, files={"file": (path.name, f)}, params=params)
        except (OSError, httpx.HTTPError) as e:
            print(f"{path}: lỗi - {e}", file=sys.stderr)
            failed += 1
            continue
        if r.is_success:
            print(json.dumps(r.json(), ensure_ascii=False, indent=2))
        else:
            print(f"{path}: HTTP {r.status_code} - {r.text}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Chạy API OCR GPKD hoặc gửi file lên API")
    parser.add_argument("files", nargs="*", help="File PDF/ảnh để gửi lên API (bỏ trống: bật server)")
    parser.add_argument("--host", default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8002")))
    parser.add_argument("--reload", action="store_true", help="Server: tự nạp lại khi sửa code (dev)")
    parser.add_argument("--url", help="Client: địa chỉ API (mặc định http://127.0.0.1:<port>)")
    parser.add_argument("--text", action="store_true", help="Client: trả kèm text OCR từng trang")
    parser.add_argument("--timeout", type=float, default=600, help="Client: timeout mỗi file (giây)")
    args = parser.parse_args()
    if args.files:
        return send(args)
    serve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
