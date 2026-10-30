"""Chạy main.run_flow cho mọi file trong thư mục và ghi report thời gian theo từng trang."""

import argparse
import csv
from pathlib import Path
import sys
import time

from main import DEFAULT_OUTPUT_DIR, SUPPORTED_SUFFIXES, _write_json, run_flow
from qwen_ocr import QwenOCR


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAGE_TIMING_KEYS = ("page", "seconds", "detect_seconds", "ocr_seconds")


def run_batch(data_dir, output_dir, template=None, refine_text=True):
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
    batch_started = time.perf_counter()
    for index, source in enumerate(files, 1):
        relative = source.relative_to(data_dir)
        print(f"[{index}/{len(files)}] {relative}", file=sys.stderr)
        started = time.perf_counter()
        doc = {"file": str(relative), "status": "ok", "error": None, "pages": []}
        try:
            result = run_flow(source, output_dir / relative.with_suffix(""), template,
                              engine_factory=engine_factory, refine_text=refine_text)
            doc["pages"] = [{key: p.get(key) for key in PAGE_TIMING_KEYS} for p in result["pages"]]
            doc["timings"] = result.get("timings", {})
        except Exception as exc:  # noqa: BLE001 - ghi lỗi rồi chạy tiếp file khác
            doc.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"  Lỗi: {doc['error']}", file=sys.stderr)
        doc["total_seconds"] = round(time.perf_counter() - started, 3)
        documents.append(doc)
        for p in doc["pages"] or [dict.fromkeys(PAGE_TIMING_KEYS)]:
            rows.append({"file": doc["file"], "page": p["page"], "page_seconds": p["seconds"],
                         "detect_seconds": p["detect_seconds"], "ocr_seconds": p["ocr_seconds"],
                         "file_total_seconds": doc["total_seconds"], "status": doc["status"],
                         "error": doc["error"]})
        print(f"  {doc['total_seconds']} giây, {len(doc['pages'])} trang", file=sys.stderr)
        # Ghi report sau mỗi file để không mất dữ liệu nếu bị dừng giữa chừng.
        summary = _write_report(output_dir, rows, documents,
                                round(time.perf_counter() - batch_started, 3))

    if documents:
        print(f"Tổng kết: {summary['files']} file ({summary['failed_files']} lỗi), "
              f"{summary['pages']} trang trong {summary['wall_seconds']} giây "
              f"({summary['pages_per_minute']} trang/phút)", file=sys.stderr)
        for prefix, label in (("file", "Mỗi file"), ("page", "Mỗi trang"),
                              ("detect", "  phát hiện"), ("ocr", "  OCR")):
            print(f"  {label}: TB {summary[f'avg_{prefix}_seconds']} giây, "
                  f"min {summary[f'min_{prefix}_seconds']}, max {summary[f'max_{prefix}_seconds']}",
                  file=sys.stderr)
    print(f"Report: {output_dir / 'timing_report.csv'}, {output_dir / 'timing_report.json'}",
          file=sys.stderr)
    return documents


def _stats(values, prefix):
    values = [v for v in values if v is not None]
    return {
        f"total_{prefix}_seconds": round(sum(values), 3),
        f"avg_{prefix}_seconds": round(sum(values) / len(values), 3) if values else None,
        f"max_{prefix}_seconds": max(values, default=None),
        f"min_{prefix}_seconds": min(values, default=None),
    }


def _write_report(output_dir, rows, documents, wall_seconds):
    with open(output_dir / "timing_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ok = [d for d in documents if d["status"] == "ok"]
    pages = [p for d in ok for p in d["pages"]]
    summary = {
        "files": len(documents),
        "failed_files": len(documents) - len(ok),
        "pages": len(pages),
        "wall_seconds": wall_seconds,
        **_stats([d["total_seconds"] for d in ok], "file"),
        **_stats([p["seconds"] for p in pages], "page"),
        **_stats([p["detect_seconds"] for p in pages], "detect"),
        **_stats([p["ocr_seconds"] for p in pages], "ocr"),
        "load_detector_seconds": round(sum(d.get("timings", {}).get("load_detector_seconds", 0)
                                           for d in ok), 3),
        "pages_per_minute": round(60 * len(pages) / wall_seconds, 2) if wall_seconds else None,
    }
    _write_json(output_dir / "timing_report.json", {"summary": summary, "documents": documents})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=DEFAULT_DATA_DIR)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "batch")
    parser.add_argument("--template", type=Path, help="Mẫu JSON thủ công (mặc định: tự nhận diện)")
    parser.add_argument("--no-heron-text", dest="refine_text", action="store_false",
                        help="Không dùng heron tách vùng chữ; OCR trực tiếp box contour (nhanh hơn)")
    args = parser.parse_args()
    run_batch(args.data_dir, args.output_dir, args.template, args.refine_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 