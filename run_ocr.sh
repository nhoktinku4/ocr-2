#!/usr/bin/env bash
# Chạy OCR cho file/thư mục ở bất kỳ đâu trên máy.
# Đặt cạnh docker-compose.yml (thư mục cha của source/), rồi:
#   ./run_ocr.sh "/home/user/data/KV MT/2737091.pdf"
#   ./run_ocr.sh /home/user/data/KV\ MT            # cả thư mục
# Script mount thư mục chứa đầu vào vào container ở đúng đường dẫn đó (chỉ đọc),
# nên đường dẫn trên máy dùng nguyên được trong container.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Cách dùng: $0 <file PDF/ảnh hoặc thư mục>" >&2
    exit 2
fi
target=$(realpath -- "$1")
if [ ! -e "$target" ]; then
    echo "Không tìm thấy: $1" >&2
    exit 1
fi
# File thì mount thư mục cha, thư mục thì mount chính nó.
if [ -d "$target" ]; then mount_dir=$target; else mount_dir=$(dirname -- "$target"); fi

cd "$(dirname -- "$(realpath -- "$0")")"
exec docker compose run --rm --no-deps -v "$mount_dir:$mount_dir:ro" --entrypoint python api main.py "$target"
