#!/usr/bin/env bash
# Chạy OCR cho file/thư mục ở bất kỳ đâu trên máy.
# Đặt cạnh docker-compose.yml (thư mục cha của source/), rồi:
#   ./run_ocr.sh "/home/user/data/KV MT/2737091.pdf"
#   ./run_ocr.sh /home/user/data/KV\ MT            # cả thư mục
#   ./run_ocr.sh /home/user/p1.jpg /home/user/p2.jpg  # nhiều ảnh = các trang của một giấy
# Script mount thư mục chứa đầu vào vào container ở đúng đường dẫn đó (chỉ đọc),
# nên đường dẫn trên máy dùng nguyên được trong container.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Cách dùng: $0 <file PDF/ảnh hoặc thư mục> [ảnh trang tiếp theo...]" >&2
    exit 2
fi
targets=() mounts=() seen=()
for arg in "$@"; do
    target=$(realpath -- "$arg")
    if [ ! -e "$target" ]; then
        echo "Không tìm thấy: $arg" >&2
        exit 1
    fi
    targets+=("$target")
    # File thì mount thư mục cha, thư mục thì mount chính nó.
    if [ -d "$target" ]; then dir=$target; else dir=$(dirname -- "$target"); fi
    # Nhiều ảnh cùng thư mục: chỉ mount một lần (docker báo lỗi khi trùng điểm mount).
    case " ${seen[*]-} " in *" $dir "*) ;; *) seen+=("$dir"); mounts+=(-v "$dir:$dir:ro") ;; esac
done

cd "$(dirname -- "$(realpath -- "$0")")"
exec docker compose run --rm --no-deps "${mounts[@]}" --entrypoint python api main.py "${targets[@]}"
