"""Như main.py nhưng tìm vùng chữ bằng DBNet (PP-OCRv4 det, ONNX) thay cho contour."""

from functools import lru_cache
import os

import cv2
import numpy as np

from box_detection import _largest_regions, sort_boxes, subtract_boxes
import main
import page_processing


@lru_cache(maxsize=1)
def _db_detector():
    """Nạp DBNet một lần; model đi kèm gói rapidocr_onnxruntime."""
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime.ch_ppocr_det import TextDetector

    root = os.path.dirname(rapidocr_onnxruntime.__file__)
    return TextDetector({
        "model_path": os.path.join(root, "models", "ch_PP-OCRv4_det_infer.onnx"),
        "use_cuda": False, "use_dml": False,
        "intra_op_num_threads": -1, "inter_op_num_threads": -1,
        "limit_side_len": 1280, "limit_type": "max",
        "std": [0.5, 0.5, 0.5], "mean": [0.5, 0.5, 0.5],
        "thresh": 0.3, "box_thresh": 0.5, "max_candidates": 1000,
        "unclip_ratio": 1.6, "use_dilation": True, "score_mode": "fast",
    })


def _dbnet_text_regions(image, top=50, crop_ratio=0.015, padding=15, gap_x=30, gap_y=20,
                        largest_boxes=None, exclude_boxes=None):
    """Cùng giao diện với box_detection._text_regions: mỗi dòng DBNet tìm được
    là một block, loại bảng khỏi box cuối cùng."""
    if largest_boxes is not None and largest_boxes <= 0:
        raise ValueError("largest_boxes phải là số nguyên dương")
    h, w = image.shape[:2]
    excluded = []
    for x1, y1, x2, y2 in (() if exclude_boxes is None else exclude_boxes):
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            excluded.append((x1, y1, x2, y2))
    detection_image = image.copy() if excluded else image
    for x1, y1, x2, y2 in excluded:
        detection_image[y1:y2, x1:x2] = 255
    cy, cx = int(h * crop_ratio), int(w * crop_ratio)
    cropped = image[cy:h - cy, cx:w - cx]
    quads, _ = _db_detector()(detection_image[cy:h - cy, cx:w - cx])
    if quads is None or len(quads) == 0:
        return cropped, (cx, cy), None
    boxes = []
    for quad in quads:
        x, y, bw, bh = cv2.boundingRect(np.asarray(quad, dtype=np.int32))
        boxes.append((x, y, x + bw, y + bh))
    # Không gộp: mỗi dòng DBNet tìm được là một block (gap_x/gap_y bỏ qua).
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:top]
    if largest_boxes is not None and not excluded:
        boxes = _largest_regions(boxes, largest_boxes)
    region_boxes = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1 + cx - padding), max(0, y1 + cy - padding)
        x2, y2 = min(w, x2 + cx + padding), min(h, y2 + cy + padding)
        region_boxes.extend(subtract_boxes((x1, y1, x2, y2), excluded))
    if largest_boxes is not None and excluded:
        region_boxes = _largest_regions(region_boxes, largest_boxes)
    blocks = [{
        "order": order, "box": [int(x1), int(y1), int(x2), int(y2)],
        "width": int(x2 - x1), "height": int(y2 - y1),
        "crop": image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None,
    } for order, (x1, y1, x2, y2) in enumerate(sort_boxes(region_boxes), 1)]
    return cropped, (cx, cy), blocks


# page_processing gọi _text_regions theo tên đã import; thay tại đó, giữ nguyên các file cũ.
page_processing._text_regions = _dbnet_text_regions
# Kết quả riêng, không ghi đè output_flow/ của main.py.
main.DEFAULT_OUTPUT_DIR = main.DEFAULT_OUTPUT_DIR.with_name("output_flow_2")


if __name__ == "__main__":
    raise SystemExit(main.main())
