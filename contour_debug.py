"""Lưu ảnh từng bước xử lý để lấy contour (giống _text_regions trong box_detection.py)."""

import argparse
import os

import cv2
import numpy as np

from box_detection import merge_boxes


def save_contour_steps(image, out_dir="output_contour_steps", crop_ratio=0.015, top=20,
                       padding=15, gap_x=30, gap_y=20, exclude_boxes=None):
    os.makedirs(out_dir, exist_ok=True)
    step = [0]

    def save(name, img):
        step[0] += 1
        path = os.path.join(out_dir, f"{step[0]:02d}_{name}.png")
        cv2.imwrite(path, img)
        print("Đã lưu:", path)

    h, w = image.shape[:2]
    save("original", image)

    excluded = []
    for x1, y1, x2, y2 in (exclude_boxes or ()):
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            excluded.append((x1, y1, x2, y2))
    detection_image = image.copy()
    for x1, y1, x2, y2 in excluded:
        detection_image[y1:y2, x1:x2] = 255
    if excluded:
        save("excluded_whiteout", detection_image)

    cy, cx = int(h * crop_ratio), int(w * crop_ratio)
    detection_crop = detection_image[cy:h - cy, cx:w - cx]
    save("cropped", detection_crop)

    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 15))
    square = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    gray = cv2.cvtColor(detection_crop, cv2.COLOR_BGR2GRAY)
    save("gray", gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    save("blur", gray)

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)
    save("blackhat", blackhat)

    gradient = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    minimum, maximum = gradient.min(), gradient.max()
    if maximum > minimum:
        gradient = 255 * ((gradient - minimum) / (maximum - minimum))
    gradient = gradient.astype("uint8")
    save("sobel_gradient", gradient)

    gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, rect)
    save("gradient_close", gradient)

    mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    save("otsu_threshold", mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, square)
    save("mask_close", mask)

    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    save("mask_dilate", mask)

    for x1, y1, x2, y2 in excluded:
        mx1, my1 = max(0, x1 - cx), max(0, y1 - cy)
        mx2, my2 = min(mask.shape[1], x2 - cx), min(mask.shape[0], y2 - cy)
        if mx2 > mx1 and my2 > my1:
            mask[my1:my2, mx1:mx2] = 0
    if excluded:
        save("mask_excluded", mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vis = detection_crop.copy()
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
    save("all_contours", vis)
    if not contours or hierarchy is None:
        return

    outer = sorted((c for i, c in enumerate(contours) if hierarchy[0][i][3] == -1),
                   key=lambda c: cv2.arcLength(c, True), reverse=True)[:top]
    vis = detection_crop.copy()
    cv2.drawContours(vis, outer, -1, (0, 255, 0), 2)
    save("outer_contours", vis)

    # Cắt ảnh theo từng contour ở bước 12 (lấy từ ảnh gốc, chưa tô trắng bảng).
    crop_dir = os.path.join(out_dir, f"{step[0]:02d}_outer_contours_crops")
    os.makedirs(crop_dir, exist_ok=True)
    src_crop = image[cy:h - cy, cx:w - cx]
    for i, c in enumerate(outer, 1):
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.imwrite(os.path.join(crop_dir, f"{i:02d}_x{x + cx}_y{y + cy}.png"),
                    src_crop[y:y + bh, x:x + bw])
    print(f"Đã lưu {len(outer)} ảnh cắt vào:", crop_dir)

    boxes = [(x, y, x + bw, y + bh) for x, y, bw, bh in map(cv2.boundingRect, outer)]
    vis = detection_crop.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    save("bounding_boxes", vis)

    merged = merge_boxes(boxes, gap_x, gap_y)
    vis = image.copy()
    for x1, y1, x2, y2 in merged:
        x1, y1 = max(0, x1 + cx - padding), max(0, y1 + cy - padding)
        x2, y2 = min(w, x2 + cx + padding), min(h, y2 + cy + padding)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    save("merged_boxes", vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="File PDF/ảnh, hoặc thư mục chứa nhiều file")
    parser.add_argument("-o", "--out-dir", default="output_contour_steps")
    args = parser.parse_args()

    from pathlib import Path

    from image_io import iter_images

    suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    src = Path(args.input).expanduser()
    files = (sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
             if src.is_dir() else [src])
    if not files:
        raise SystemExit(f"Không có file PDF/ảnh trong: {src}")
    for file in files:
        print(f"=== {file}")
        try:
            for name, img in iter_images(file):
                save_contour_steps(img, os.path.join(args.out_dir, name))
        except Exception as exc:
            print(f"Lỗi khi xử lý {file}: {exc}")
