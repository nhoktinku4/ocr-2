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
 

def upscale(image, min_long=2000):
    """Phóng ảnh nhỏ (chụp màn hình, ảnh nén) cho cạnh dài >= min_long."""
    scale = min_long / max(image.shape[:2])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.05 else image


def skew_angle(image, max_deg=8, step=0.25):
    """Thử xoay; góc cho tổng điểm đen theo hàng 'nhọn' nhất là góc dòng chữ thẳng."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1000 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h, w = bw.shape
    best, best_angle = -1, 0.0
    for angle in np.arange(-max_deg, max_deg + step, step):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rows = cv2.warpAffine(bw, matrix, (w, h), flags=cv2.INTER_NEAREST).sum(axis=1, dtype=np.float64)
        score = np.square(np.diff(rows)).sum()
        if score > best:
            best, best_angle = score, angle
    return float(best_angle)


def deskew(image, min_deg=0.3):
    angle = skew_angle(image)
    if abs(angle) < min_deg:
        return image, 0.0
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE), angle


def remove_background(image):
    """max(B, G, R) xóa hoa văn nền màu, chia cho nền ước lượng để cân bằng sáng,
    rồi kéo giãn tương phản. Chữ màu (tiêu đề đỏ, dấu mộc) sẽ nhạt đi.
    Trả về ảnh BGR 3 kênh (xám) cùng kích thước."""
    gray = image.max(axis=2)
    k = max(15, (min(gray.shape) // 40) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    normalized = cv2.divide(gray, cv2.GaussianBlur(background, (k, k), 0), scale=255)
    low = np.percentile(normalized, 1)
    out = (normalized.astype(np.float32) - low) * 255 / max(1, 245 - low)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def sharpen(image, amount=0.6):
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def prepare_page(image):
    """Phóng ảnh nhỏ -> xoay thẳng -> xóa nền, cân bằng sáng -> làm nét. Trả về (ảnh, góc đã xoay)."""
    image, angle = deskew(upscale(image))
    return sharpen(remove_background(image)), angle
