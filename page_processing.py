#!/usr/bin/env python3
"""Xử lý một trang ảnh: phát hiện vùng chữ/bảng, lưu crop và tạo báo cáo."""

import json
import time
from pathlib import Path

import cv2

from box_detection import _text_regions
from ocr_image_preparation import clean_image, prepare_page, resize_large_image
from table_detection import (LAYOUT_MODEL, PADDING, TABLE_PADDING, detect_table_regions,
                             load_detector, refine_text_blocks)


def _timer(timings):
    def timed(name, function, *args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        timings[name] = round((time.perf_counter() - start) * 1000, 2)
        return result
    return timed


def _build_report(name, original, cleaned, timings, blocks, *, table_blocks=None, **extra):
    h, w = original.shape[:2]
    ch, cw = cleaned.shape[:2]
    report = {
        "name": name, "original_size": [w, h], "cleaned_size": [cw, ch],
        "margin_removed_pct": round(100 * (1 - ch * cw / (h * w)), 2),
        **extra,
        "text_block_count": len(blocks), "timings_ms": timings,
    }
    for key, regions in (("table_blocks", table_blocks), ("text_blocks", blocks)):
        if regions is not None:
            report[key] = [{k: v for k, v in block.items() if k != "crop"} for block in regions]
    return report


def draw_boxes(image, text_blocks, table_blocks):
    """Vẽ box lên bản sao ảnh: vùng chữ xanh dương, bảng xanh lá, kèm số thứ tự."""
    canvas = image.copy()
    thickness = max(2, round(max(canvas.shape[:2]) / 600))
    for regions, color, prefix in ((text_blocks, (255, 0, 0), "T"), (table_blocks, (0, 180, 0), "B")):
        for block in regions:
            x1, y1, x2, y2 = block["box"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(canvas, f"{prefix}{block['order']}", (x1 + 4, max(y1 - 6, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, thickness * 0.4, color, thickness)
    return canvas


def process_single_image(image_bgr, stem_name, output_dir,
                         crop_blocks=True, padding=PADDING, save=False, largest_boxes=None,
                         *, detector=None, model_path=LAYOUT_MODEL, iou_threshold=0.1,
                         conf=0.25, device=None, table_padding=TABLE_PADDING,
                         include_crops=False, refine_text=True):
    """Phát hiện tất cả bảng trước khi lấy các khối chữ không giao với bảng.

    largest_boxes chỉ giới hạn vùng chữ ngoài bảng; mọi bảng đều được loại trừ.
    table_padding chỉ mở rộng crop bảng để OCR; vùng tô trắng dùng box không padding.
    refine_text=False bỏ bước heron tách vùng chữ, OCR trực tiếp box contour.
    """
    if padding < 0 or table_padding < 0:
        raise ValueError("padding/table_padding phải >= 0.")
    if largest_boxes is not None and largest_boxes <= 0:
        raise ValueError("largest_boxes phải là số nguyên dương.")
    if detector is None:
        detector = load_detector(model_path, device)
    output = Path(output_dir) / stem_name
    timings = {}
    timed = _timer(timings)
    resized = timed("resize_ms", resize_large_image, image_bgr)
    prepared, skew_angle = timed("prepare_ms", prepare_page, resized)
    cleaned = timed("clean_ms", clean_image, prepared, padding=padding)
    detected_count, table_blocks = timed(
        "table_detect_ms", detect_table_regions, cleaned, detector, padding=table_padding,
        iou_threshold=iou_threshold, conf=conf,
    )
    _, _, blocks = timed(
        "contour_ms", _text_regions, cleaned, padding=padding,
        largest_boxes=largest_boxes, exclude_boxes=[block["box"] for block in table_blocks],
    )
    blocks = blocks or []
    if refine_text:
        blocks = timed("text_detect_ms", refine_text_blocks, cleaned, blocks, detector, conf=conf)
    if save:
        output.mkdir(parents=True, exist_ok=True)
        preview = output / "boxes.png"
        if not cv2.imwrite(str(preview), draw_boxes(cleaned, blocks, table_blocks)):
            raise OSError(f"Không thể lưu ảnh: {preview}")
        if crop_blocks:
            for folder, prefix, regions in (("crops", "block", blocks), ("tables", "table", table_blocks)):
                if folder == "tables" and not regions:
                    continue
                (output / folder).mkdir(exist_ok=True)
                for block in regions:
                    path = output / folder / f"{prefix}_{block['order']:03d}.png"
                    if not cv2.imwrite(str(path), block["crop"]):
                        raise OSError(f"Không thể lưu ảnh: {path}")
    h, w = image_bgr.shape[:2]
    ch, cw = cleaned.shape[:2]
    print(f"[*] {stem_name}: {w}x{h} → {cw}x{ch}, heron {detected_count} bảng thô"
          f" → {len(table_blocks)} bảng, {len(blocks)} vùng chữ ngoài bảng."
          + (f" Kết quả: {output}" if save else ""))
    report = _build_report(
        stem_name, image_bgr, cleaned, timings, blocks,
        coordinate_space="cleaned", box_format="xyxy", skew_angle=skew_angle, detected_box_count=detected_count,
        table_block_count=len(table_blocks), table_blocks=table_blocks,
    )
    if save:
        (output / "boxes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if include_crops:
        report["text_blocks"] = blocks
        report["table_blocks"] = table_blocks
    return report
 