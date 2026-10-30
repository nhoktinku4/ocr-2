"""Chạy main.run_flow cho mọi file trong thư mục và ghi report thời gian theo từng trang."""

import argparse
import csv
from pathlib import Path
import sys
import time

from main import DEFAULT_OUTPUT_DIR, SUPPORTED_SUFFIXES, _write_json, run_flow
from qwen_ocr import QwenOCR


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run_batch(data_dir, output_dir, template=None):
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    print(f"Tìm thấy {len(files)} file trong {data_dir}", file=sys.stderr)

    # Nạp model OCR một lần, dùng chung cho mọi file.
    engine = None

    def engine_factory():
        nonlocal engine
        if engine is None:
            engine = QwenOCR()
        return engine

    rows, documents = [], []
    for index, source in enumerate(files, 1):
        relative = source.relative_to(data_dir)
        print(f"[{index}/{len(files)}] {relative}", file=sys.stderr)
        started = time.perf_counter()
        doc = {"file": str(relative), "status": "ok", "error": None, "pages": []}
        try:
            result = run_flow(source, output_dir / relative.with_suffix(""), template,
                              engine_factory=engine_factory)
            doc["pages"] = [{"page": p["page"], "seconds": p["seconds"]} for p in result["pages"]]
        except Exception as exc:  # noqa: BLE001 - ghi lỗi rồi chạy tiếp file khác
            doc.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"  Lỗi: {doc['error']}", file=sys.stderr)
        doc["total_seconds"] = round(time.perf_counter() - started, 3)
        documents.append(doc)
        for p in doc["pages"] or [{"page": None, "seconds": None}]:
            rows.append({"file": doc["file"], "page": p["page"], "page_seconds": p["seconds"],
                         "file_total_seconds": doc["total_seconds"], "status": doc["status"],
                         "error": doc["error"]})
        # Ghi report sau mỗi file để không mất dữ liệu nếu bị dừng giữa chừng.
        _write_report(output_dir, rows, documents)

    print(f"Report: {output_dir / 'timing_report.csv'}", file=sys.stderr)
    return documents


def _write_report(output_dir, rows, documents):
    with open(output_dir / "timing_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    page_times = [r["page_seconds"] for r in rows if r["page_seconds"] is not None]
    summary = {
        "files": len(documents),
        "failed_files": sum(d["status"] != "ok" for d in documents),
        "pages": len(page_times),
        "total_page_seconds": round(sum(page_times), 3),
        "avg_page_seconds": round(sum(page_times) / len(page_times), 3) if page_times else None,
        "max_page_seconds": max(page_times, default=None),
        "min_page_seconds": min(page_times, default=None),
    }
    _write_json(output_dir / "timing_report.json", {"summary": summary, "documents": documents})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=DEFAULT_DATA_DIR)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "batch")
    parser.add_argument("--template", type=Path, help="Mẫu JSON thủ công (mặc định: tự nhận diện)")
    args = parser.parse_args()
    run_batch(args.data_dir, args.output_dir, args.template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 