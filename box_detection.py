"""Tìm vùng chữ và thao tác trên các box."""

import cv2
import numpy as np


def _largest_regions(boxes, limit):
    return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:limit]


def sort_boxes(boxes, y_threshold=20):
    lines = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        if lines:
            prev = lines[-1][-1]
            threshold = min(prev[3] - prev[1], box[3] - box[1]) * 0.5
            overlap = min(box[3], prev[3]) - max(box[1], prev[1])
            if overlap > threshold or abs(box[1] - prev[1]) < max(y_threshold, threshold):
                lines[-1].append(box)
                continue
        lines.append([box])
    return [box for line in lines for box in sorted(line, key=lambda b: b[0])]


def merge_boxes(boxes, gap_x=30, gap_y=20):
    boxes = boxes.copy()
    changed = True
    while changed:
        changed, merged, used = False, [], set()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if i in used:
                continue
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                a1, b1, a2, b2 = boxes[j]
                dx, dy = max(0, a1 - x2, x1 - a2), max(0, b1 - y2, y1 - b2)
                overlap_x, overlap_y = min(x2, a2) - max(x1, a1), min(y2, b2) - max(y1, b1)
                if (dx == 0 and dy == 0) or (dx <= gap_x and overlap_y > 0) or (dy <= gap_y and overlap_x > 0):
                    x1, y1, x2, y2 = min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)
                    used.add(j)
                    changed = True
            merged.append((x1, y1, x2, y2))
        boxes = merged
    return boxes


def subtract_boxes(box, excluded_boxes):
    """Chia hình chữ nhật thành các phần không giao với bất kỳ box loại trừ nào."""
    remaining = [tuple(box)]
    for ex1, ey1, ex2, ey2 in excluded_boxes:
        pieces = []
        for x1, y1, x2, y2 in remaining:
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            if ix1 >= ix2 or iy1 >= iy2:
                pieces.append((x1, y1, x2, y2))
                continue
            for part in ((x1, y1, x2, iy1), (x1, iy2, x2, y2),
                         (x1, iy1, ix1, iy2), (ix2, iy1, x2, iy2)):
                if part[2] > part[0] and part[3] > part[1]:
                    pieces.append(part)
        remaining = pieces
    return remaining


def _text_regions(image, top=50, crop_ratio=0.015, padding=15, gap_x=30, gap_y=20,
                  largest_boxes=None, exclude_boxes=None):
    """Tìm vùng chữ; exclude_boxes loại bảng khỏi cả mask và box cuối cùng."""
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
    detection_crop = detection_image[cy:h - cy, cx:w - cx]
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 15))
    square = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    gray = cv2.GaussianBlur(cv2.cvtColor(detection_crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)
    gradient = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    minimum, maximum = gradient.min(), gradient.max()
    if maximum > minimum:
        gradient = 255 * ((gradient - minimum) / (maximum - minimum))
    gradient = cv2.morphologyEx(gradient.astype("uint8"), cv2.MORPH_CLOSE, rect)
    mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, square)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    # Morphology có thể lan trở lại vùng bảng; loại vùng này khỏi mask lần nữa.
    for x1, y1, x2, y2 in excluded:
        mx1, my1 = max(0, x1 - cx), max(0, y1 - cy)
        mx2, my2 = min(mask.shape[1], x2 - cx), min(mask.shape[0], y2 - cy)
        if mx2 > mx1 and my2 > my1:
            mask[my1:my2, mx1:mx2] = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return cropped, (cx, cy), None
    contours = sorted((c for i, c in enumerate(contours) if hierarchy[0][i][3] == -1),
                      key=lambda c: cv2.arcLength(c, True), reverse=True)[:top]
    boxes = [(x, y, x + bw, y + bh) for x, y, bw, bh in map(cv2.boundingRect, contours)]
    boxes = merge_boxes(boxes, gap_x, gap_y)
    if largest_boxes is not None and not excluded:
        # Chọn theo diện tích sau khi gộp, trước khi thêm padding.
        boxes = _largest_regions(boxes, largest_boxes)
    region_boxes = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1 + cx - padding), max(0, y1 + cy - padding)
        x2, y2 = min(w, x2 + cx + padding), min(h, y2 + cy + padding)
        # Cả box gộp lẫn padding đều phải nằm hoàn toàn ngoài bảng.
        for part in subtract_boxes((x1, y1, x2, y2), excluded):
            px1, py1, px2, py2 = part
            if excluded:
                mx1, my1 = max(0, px1 - cx), max(0, py1 - cy)
                mx2, my2 = min(mask.shape[1], px2 - cx), min(mask.shape[0], py2 - cy)
                if mx2 <= mx1 or my2 <= my1 or not np.any(mask[my1:my2, mx1:mx2]):
                    continue
            region_boxes.append(part)
    if largest_boxes is not None and excluded:
        region_boxes = _largest_regions(region_boxes, largest_boxes)
    blocks = [{
        "order": order, "box": [int(x1), int(y1), int(x2), int(y2)],
        "width": int(x2 - x1), "height": int(y2 - y1),
        "crop": image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None,
    } for order, (x1, y1, x2, y2) in enumerate(sort_boxes(region_boxes), 1)]
    return cropped, (cx, cy), blocks
 