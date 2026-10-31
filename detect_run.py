"""Chỉ phát hiện vùng chữ/bảng (không OCR) cho một file hoặc cả thư mục PDF/ảnh.

Các bước giống page_processing.process_single_image:
    resize → chỉnh nghiêng → làm sạch → heron tìm bảng → contour lấy khung chữ ngoài bảng
rồi chạy heron trên từng khung contour, tách box bao nhiều dòng thành từng dòng, gộp box trùng/chồng nhau (IoU hoặc box nằm trong box
khác), gộp box gần nhau trên cùng dòng và bỏ box chữ nằm trong bảng. Kết quả mỗi trang: boxes.png, boxes.json, crops/, tables/.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from box_detection import _text_regions, sort_boxes
from image_io import iter_images
from ocr_image_preparation import clean_image, prepare_page, resize_large_image
from page_processing import _build_report, _timer, draw_boxes
from table_detection import PADDING, TABLE_PADDING, TEXT_LABELS, detect_table_regions, load_detector


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output_detect"
# Như main.SUPPORTED_SUFFIXES; không import main để khỏi nạp phần OCR.
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LAYOUT_LABELS = TEXT_LABELS | {"key_value_region"}


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _spans_rows(boxes):
    """True nếu có ít nhất 2 box nằm ở 2 dòng khác nhau (không chồng theo trục y)."""
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if min(a[3], b[3]) - max(a[1], b[1]) < 0.5 * min(a[3] - a[1], b[3] - b[1]):
                return True
    return False


def _uncovered_rows(box, inner):
    """Các dải ngang của box bao cụm chưa có box con nào phủ (vd dòng giữa heron chỉ bắt được trong box bao)."""
    min_height = 0.6 * min(o[3] - o[1] for o in inner)
    rows, y = [], box[1]
    for oy1, oy2 in sorted((o[1], o[3]) for o in inner):
        if oy1 - y >= min_height:
            rows.append((box[0], y, box[2], oy1))
        y = max(y, oy2)
    if box[3] - y >= min_height:
        rows.append((box[0], y, box[2], box[3]))
    return rows


def _union_area(boxes, frame):
    """Diện tích phần hợp của boxes trong khung frame (box con chồng nhau không bị tính 2 lần)."""
    mask = np.zeros((frame[3] - frame[1], frame[2] - frame[0]), bool)
    for x1, y1, x2, y2 in boxes:
        mask[max(0, y1 - frame[1]):max(0, y2 - frame[1]), max(0, x1 - frame[0]):max(0, x2 - frame[0])] = True
    return int(mask.sum())


def split_containers(boxes, contain_thr=0.8):
    """Thay box bao cụm (chứa >= 2 box con ở nhiều dòng) bằng các dải dòng chưa có box con phủ."""
    kept = []
    for b in boxes:
        # Chỉ xét box con nhỏ hơn hẳn; box gần trùng b (heron hay trả 2 lần) không tính.
        inner = [o for o in boxes if o is not b and _inter(o, b) >= contain_thr * _area(o)
                 and _area(o) < 0.8 * _area(b)]
        # Box con phải phủ phần đáng kể của b; nếu chỉ là vài mẩu nhỏ (vd nhãn bên trái
        # của dòng có giá trị xuống dòng) thì b là một dòng thật, giữ nguyên.
        covers = bool(inner) and _union_area(inner, b) >= 0.3 * _area(b)
        kept += _uncovered_rows(b, inner) if covers and _spans_rows(inner) else [b]
    return kept


def merge_boxes(boxes, iou_thr=1.0, contain_thr=0.8):
    """Gộp box trùng/chồng nhau (IoU >= iou_thr hoặc giao / box nhỏ >= contain_thr), lặp đến khi ổn định."""
    changed = True
    while changed:
        changed, out = False, []
        for box in sorted(boxes, key=_area, reverse=True):
            for i, m in enumerate(out):
                inter = _inter(box, m)
                if inter and (inter / max(_area(box) + _area(m) - inter, 1) >= iou_thr
                              or inter / max(min(_area(box), _area(m)), 1) >= contain_thr):
                    out[i] = (min(box[0], m[0]), min(box[1], m[1]), max(box[2], m[2]), max(box[3], m[3]))
                    changed = True
                    break
            else:
                out.append(box)
        boxes = out
    return boxes


def merge_horizontal(boxes, gap_ratio=2.0):
    """Gộp box cùng dòng nằm gần nhau theo chiều ngang (vd nhãn "Sinh ngày:" + giá trị "01/09/1987").

    Cùng dòng: chồng theo trục y >= 50% box thấp hơn. Gần: khoảng trống ngang <= gap_ratio × chiều cao box thấp hơn.
    """
    changed = True
    while changed:
        changed, out = False, []
        for box in sorted(boxes, key=lambda b: b[0]):
            for i, m in enumerate(out):
                height = min(box[3] - box[1], m[3] - m[1])
                same_row = min(box[3], m[3]) - max(box[1], m[1]) >= 0.5 * height
                gap = max(box[0], m[0]) - min(box[2], m[2])
                if same_row and gap <= gap_ratio * height:
                    out[i] = (min(box[0], m[0]), min(box[1], m[1]), max(box[2], m[2]), max(box[3], m[3]))
                    changed = True
                    break
            else:
                out.append(box)
        boxes = out
    return boxes


def drop_text_in_tables(boxes, tables, contain_thr=0.8):
    """Bỏ box chữ có >= contain_thr diện tích nằm trong một bảng."""
    return [b for b in boxes
            if not any(_inter(b, t) >= contain_thr * max(_area(b), 1) for t in tables)]


def layout_text_blocks(image, blocks, detector, conf=0.3, split=True, merge=True, hmerge_gap=2.0, iou_thr=1.0, contain_thr=0.8,
                       tables=(), table_contain_thr=0.8):
    """Chạy heron trên từng khung contour, gộp box trùng, bỏ box nằm trong bảng; tọa độ theo ảnh image."""
    h, w = image.shape[:2]
    boxes = []
    for block in blocks:
        bx1, by1, bx2, by2 = block["box"]
        boxes += [(max(0, math.floor(x1) + bx1), max(0, math.floor(y1) + by1),
                   min(w, math.ceil(x2) + bx1), min(h, math.ceil(y2) + by1))
                  for x1, y1, x2, y2 in detector(image[by1:by2, bx1:bx2], conf=conf, labels=LAYOUT_LABELS)]
    if split:
        boxes = split_containers(boxes, contain_thr)
    if merge:
        boxes = merge_boxes(boxes, iou_thr, contain_thr)
    if hmerge_gap > 0:
        boxes = merge_horizontal(boxes, hmerge_gap)
    boxes = drop_text_in_tables(boxes, tables, table_contain_thr)
    return [{"order": order, "box": [x1, y1, x2, y2], "width": x2 - x1, "height": y2 - y1,
             "crop": image[y1:y2, x1:x2]}
            for order, (x1, y1, x2, y2) in enumerate(sort_boxes(boxes), 1)
            if x2 > x1 and y2 > y1]


def detect_page(image_bgr, name, output, detector, *, padding=PADDING, table_padding=TABLE_PADDING,
                conf=0.25, text_conf=0.3, crop_blocks=True, **post_kwargs):
    timings = {}
    timed = _timer(timings)
    resized = timed("resize_ms", resize_large_image, image_bgr)
    prepared, skew_angle = timed("prepare_ms", prepare_page, resized)
    cleaned = timed("clean_ms", clean_image, prepared, padding=padding)
    detected_count, table_blocks = timed("table_detect_ms", detect_table_regions, cleaned, detector,
                                         padding=table_padding, conf=conf)
    _, _, contour_blocks = timed("contour_ms", _text_regions, cleaned, padding=padding,
                                 exclude_boxes=[b["box"] for b in table_blocks])
    blocks = timed("text_detect_ms", layout_text_blocks, cleaned, contour_blocks or [], detector,
                   conf=text_conf, tables=[b["box"] for b in table_blocks], **post_kwargs)

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
        contour_blocks=[b["box"] for b in contour_blocks or []],
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
    parser.add_argument("--text-conf", type=float, default=0.3, help="Ngưỡng heron cho vùng chữ")
    parser.add_argument("--no-split", action="store_true",
                        help="Không tách box bao nhiều dòng trước khi gộp (giữ khối lớn)")
    parser.add_argument("--hmerge-gap", type=float, default=2.0,
                        help="Gộp box cùng dòng có khoảng trống ngang <= hệ số × chiều cao dòng (0 = tắt)")
    parser.add_argument("--no-merge", action="store_true", help="Tắt gộp box trùng")
    parser.add_argument("--merge-iou", type=float, default=1.0, help="Ngưỡng IoU để gộp")
    parser.add_argument("--merge-contain", type=float, default=0.8,
                        help="Ngưỡng giao / diện tích box nhỏ để gộp (box nằm trong box khác)")
    parser.add_argument("--table-contain", type=float, default=0.8,
                        help="Tỉ lệ diện tích box chữ nằm trong bảng để bị bỏ")
    parser.add_argument("--no-crops", dest="crop_blocks", action="store_false", help="Không lưu ảnh crop")
    args = parser.parse_args()
    run(args.source, args.output_dir, text_conf=args.text_conf, crop_blocks=args.crop_blocks,
        split=not args.no_split, merge=not args.no_merge, hmerge_gap=args.hmerge_gap, iou_thr=args.merge_iou, contain_thr=args.merge_contain,
        table_contain_thr=args.table_contain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
