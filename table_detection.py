"""Nạp docling-layout-heron và phát hiện vùng bảng."""

import math
from pathlib import Path

import cv2

from box_detection import _largest_regions, sort_boxes


LAYOUT_MODEL = Path(__file__).resolve().parent / "docling-layout-heron"
PADDING = 30
TABLE_PADDING = 30
# Nhãn heron coi là vùng chữ khi tách box chữ thành vùng nhỏ hơn.
TEXT_LABELS = frozenset({"caption", "footnote", "list_item", "page_footer", "page_header",
                         "section_header", "text", "title"})


class HeronDetector:
    """Bọc processor + model RT-DETRv2 của docling-layout-heron."""

    def __init__(self, model_path, device=None):
        try:
            import torch
            from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
        except ImportError as exc:
            raise RuntimeError("Thiếu transformers/torch; hãy dùng môi trường ocr hoặc cài "
                               "pip install transformers torch.") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(str(model_path))
        self.model = RTDetrV2ForObjectDetection.from_pretrained(str(model_path)).to(self.device).eval()

    def __call__(self, image_bgr, conf=0.25, labels=frozenset({"table"})):
        """Trả về các box thuộc labels dạng (x1, y1, x2, y2) float theo pixel ảnh gốc."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        h, w = image_bgr.shape[:2]
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=conf,
        )[0]
        names = self.model.config.id2label
        return [tuple(map(float, box)) for box, label in zip(result["boxes"], result["labels"])
                if names[int(label)].lower() in labels]


def load_detector(model_path=LAYOUT_MODEL, device=None):
    model_path = Path(model_path).expanduser().resolve()
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Không tìm thấy model docling-layout-heron: {model_path}.")
    return HeronDetector(model_path, device)


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def merge_table_boxes(boxes, iou_threshold=0.1):
    """Gộp lặp đến khi ổn định; chỉ gộp khi IoU lớn hơn ngưỡng, như file mẫu."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold phải nằm trong khoảng 0 đến 1.")
    boxes = list(boxes)
    while True:
        changed, merged = False, []
        while boxes:
            x1, y1, x2, y2 = boxes.pop(0)
            index = 0
            while index < len(boxes):
                other = boxes[index]
                if compute_iou((x1, y1, x2, y2), other) > iou_threshold:
                    x1, y1 = min(x1, other[0]), min(y1, other[1])
                    x2, y2 = max(x2, other[2]), max(y2, other[3])
                    boxes.pop(index)
                    changed = True
                else:
                    index += 1
            merged.append((x1, y1, x2, y2))
        boxes = merged
        if not changed:
            return boxes


def detect_table_regions(image, detector, padding=TABLE_PADDING, iou_threshold=0.1,
                         largest_boxes=None, conf=0.25):
    """Box dùng để che bảng không có padding; chỉ crop OCR được mở rộng."""
    if padding < 0 or (largest_boxes is not None and largest_boxes <= 0):
        raise ValueError("padding phải >= 0; largest_boxes phải là số nguyên dương.")
    if not 0 <= conf <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("conf/iou_threshold phải nằm trong khoảng 0 đến 1.")
    # Làm tròn ra ngoài để không bỏ sót mép bảng khi tạo mask nguyên pixel.
    boxes = [(math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
             for x1, y1, x2, y2 in detector(image, conf=conf)]
    detected_count = len(boxes)
    boxes = merge_table_boxes(boxes, iou_threshold)
    if largest_boxes is not None:
        boxes = _largest_regions(boxes, largest_boxes)
    h, w = image.shape[:2]
    blocks = []
    for x1, y1, x2, y2 in sort_boxes(boxes):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
            cx2, cy2 = min(w, x2 + padding), min(h, y2 + padding)
            blocks.append({
                "order": len(blocks) + 1, "box": [x1, y1, x2, y2],
                "width": x2 - x1, "height": y2 - y1,
                "crop_box": [cx1, cy1, cx2, cy2],
                "crop": image[cy1:cy2, cx1:cx2],
            })
    return detected_count, blocks


def _drop_contained(boxes, ratio=0.7):
    """Bỏ box có ít nhất ratio diện tích nằm trong box lớn hơn để không OCR trùng."""
    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    kept = []
    for box in sorted(boxes, key=area, reverse=True):
        inside = any(area((max(box[0], k[0]), max(box[1], k[1]),
                           min(box[2], k[2]), min(box[3], k[3]))) >= ratio * area(box) for k in kept)
        if not inside:
            kept.append(box)
    return kept


def refine_text_blocks(image, blocks, detector, padding=10, iou_threshold=0.1, conf=0.25):
    """Chạy heron trên từng box chữ; chỉ giữ vùng chữ model tìm được.

    Box contour không có vùng chữ nào (mép giấy, viền bảng) bị bỏ.
    """
    h, w = image.shape[:2]
    boxes = []
    for block in blocks:
        bx1, by1, bx2, by2 = block["box"]
        found = [(math.floor(x1) + bx1, math.floor(y1) + by1, math.ceil(x2) + bx1, math.ceil(y2) + by1)
                 for x1, y1, x2, y2 in detector(image[by1:by2, bx1:bx2], conf=conf, labels=TEXT_LABELS)]
        for x1, y1, x2, y2 in merge_table_boxes(found, iou_threshold):
            # Thêm padding nhưng không vượt khỏi box chữ gốc.
            boxes.append((max(bx1, x1 - padding), max(by1, y1 - padding),
                          min(bx2, x2 + padding), min(by2, y2 + padding)))
    refined = []
    for x1, y1, x2, y2 in sort_boxes(_drop_contained(boxes)):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            refined.append({
                "order": len(refined) + 1, "box": [x1, y1, x2, y2],
                "width": x2 - x1, "height": y2 - y1, "crop": image[y1:y2, x1:x2],
            })
    return refined
