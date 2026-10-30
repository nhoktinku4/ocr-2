"""Chuẩn bị ảnh trang và ảnh cắt trước khi đưa vào OCR."""

import math

import cv2
import numpy as np
from PIL import Image


MAX_PREPROCESS_PIXELS = 40_000_000
MAX_OCR_PIXELS = 1_605_632


def resize_large_image(cv2_img, max_pixels=MAX_PREPROCESS_PIXELS):
    h, w = cv2_img.shape[:2]
    if h * w <= max_pixels:
        return cv2_img
    scale = np.sqrt(max_pixels / (h * w))
    new_w = min(max_pixels, max(1, int(w * scale)))
    new_h = min(max_pixels // new_w, max(1, int(h * scale)))
    return cv2.resize(cv2_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def enhance_ocr_crop(image, *, max_pixels=MAX_OCR_PIXELS):
    """Cải thiện ảnh cắt cho OCR, giữ màu và giới hạn số pixel đầu vào model."""
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError("max_pixels phải là số nguyên dương.")
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if not h or not w:
        raise ValueError("Ảnh OCR không được rỗng.")
    rgb = resize_large_image(rgb, max_pixels)
    h, w = rgb.shape[:2]

    # Chỉ tăng tương phản cho ảnh nhạt; giữ nguyên sắc độ của mực và dấu mộc.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    low, high = np.percentile(lightness, (1, 99))
    if 15 <= high - low < 130:
        adjusted = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lightness)
        lab[:, :, 0] = np.clip(high - (high - adjusted.astype(np.float32)) * 1.25, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Phóng vùng nhỏ để chữ dễ đọc hơn, nhưng không vượt giới hạn ảnh của Qwen.
    if min(h, w) < 400 and h * w < max_pixels:
        scale = min(2.0, 400 / min(h, w), math.sqrt(max_pixels / (h * w)))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        if new_w > w or new_h > h:
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # Làm sắc nét nhẹ sau nội suy, tránh mất nét các dấu nhỏ.
            blurred = cv2.GaussianBlur(rgb, (0, 0), 0.8)
            rgb = cv2.addWeighted(rgb, 1.15, blurred, -0.15, 0)
    return Image.fromarray(rgb)


def clean_image(image, padding=15):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    x, y, w, h = cv2.boundingRect(mask)
    ih, iw = image.shape[:2]
    return image[max(0, y - padding):min(ih, y + h + padding),
                 max(0, x - padding):min(iw, x + w + padding)] if w and h else image
 