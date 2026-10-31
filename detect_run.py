"""Chỉ phát hiện vùng chữ/bảng (không OCR) cho một file hoặc cả thư mục PDF/ảnh.

Các bước giống page_processing.process_single_image:
    resize → chỉnh nghiêng → làm sạch → heron tìm bảng → contour lấy khung chữ ngoài bảng
Vùng chữ lấy trực tiếp từ contour (không dùng model), heron chỉ dùng để tìm bảng.
Kết quả mỗi trang: boxes.png, boxes.json, crops/, tables/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from box_detection import _text_regions
from image_io import iter_images
from ocr_image_preparation import clean_image, prepare_page, resize_large_image
from page_processing import _build_report, _timer, draw_boxes
from table_detection import PADDING, TABLE_PADDING, detect_table_regions, load_detector


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output_detect"
# Như main.SUPPORTED_SUFFIXES; không import main để khỏi nạp phần OCR.
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def detect_page(image_bgr, name, output, detector, *, padding=PADDING, table_padding=TABLE_PADDING,
                conf=0.25, crop_blocks=True):
    timings = {}
    timed = _timer(timings)
    resized = timed("resize_ms", resize_large_image, image_bgr)
    prepared, skew_angle = timed("prepare_ms", prepare_page, resized)
    cleaned = timed("clean_ms", clean_image, prepared, padding=padding)
    detected_count, table_blocks = timed("table_detect_ms", detect_table_regions, cleaned, detector,
                                         padding=table_padding, conf=conf)
    _, _, blocks = timed("contour_ms", _text_regions, cleaned, padding=padding,
                         exclude_boxes=[b["box"] for b in table_blocks])
    blocks = blocks or []

    page_dir = Path(output) / name
    page_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(page_dir / "boxes.png"), draw_boxes(cleaned, blocks, table_blocks))
    if crop_blocks:
        for folder, prefix, regions in (("crops", "block", blocks), ("tables", "table", table_blocks)):
            if regions:
                (page_dir / folder).mkdir(exist_ok=True)
            for block in regions:
                cv2.imwrite(str(page_dir / folder / f"{prefix}_{block['order']:03d}.png"), block["crop"])
    report = _build_report(
        name, image_bgr, cleaned, timings, blocks,
        coordinate_space="cleaned", box_format="xyxy", skew_angle=skew_angle,
        detected_box_count=detected_count, table_block_count=len(table_blocks), table_blocks=table_blocks,
    )
    (page_dir / "boxes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name}: {len(table_blocks)} bảng, {len(blocks)} vùng chữ", file=sys.stderr)
    return report


def run(source, output_dir, **options):
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    files = (sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
             if source.is_dir() else [source])
    detector = load_detector()
    print(f"Tìm thấy {len(files)} file", file=sys.stderr)
    summary = []
    for index, path in enumerate(files, 1):
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        print(f"[{index}/{len(files)}] {relative}", file=sys.stderr)
        started = time.perf_counter()
        doc = {"file": str(relative), "status": "ok", "error": None, "pages": 0}
        try:
            for name, image in iter_images(path):
                detect_page(image, name, output_dir / relative.with_suffix(""), detector, **options)
                doc["pages"] += 1
        except Exception as exc:  # noqa: BLE001 - ghi lỗi rồi chạy tiếp file khác
            doc.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"  Lỗi: {doc['error']}", file=sys.stderr)
        doc["seconds"] = round(time.perf_counter() - started, 3)
        summary.append(doc)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "detect_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    failed = sum(d["status"] != "ok" for d in summary)
    print(f"Xong: {len(summary) - failed}/{len(summary)} file OK. Report: {output_dir / 'detect_report.json'}",
          file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File PDF/ảnh hoặc thư mục (quét cả thư mục con)")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-crops", dest="crop_blocks", action="store_false", help="Không lưu ảnh crop")
    args = parser.parse_args()
    run(args.source, args.output_dir, crop_blocks=args.crop_blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
