"""Đọc ảnh và render từng trang PDF trong giới hạn kích thước."""

import math
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

from ocr_image_preparation import MAX_PREPROCESS_PIXELS


def iter_pdf_images(pdf_path, dpi=300, max_pixels=MAX_PREPROCESS_PIXELS):
    """Giới hạn kích thước ngay khi render, trước khi Pillow mở ảnh trang PDF."""
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi phải là số hữu hạn lớn hơn 0.")
    if not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError("max_pixels phải là số nguyên dương.")
    import pymupdf

    with pymupdf.open(str(pdf_path)) as document:
        if not document.is_pdf:
            raise ValueError(f"File không phải PDF: {pdf_path}")
        if document.needs_pass:
            raise ValueError("PDF có mật khẩu. Hãy mở khóa PDF trước khi xử lý.")
        for page_number, page in enumerate(document, 1):
            # Poppler scale theo MediaBox rồi xoay ảnh theo hướng của trang.
            width, height = page.mediabox.width, page.mediabox.height
            width, height = math.ceil(width * dpi / 72), math.ceil(height * dpi / 72)
            scale = min(1, math.sqrt(max_pixels / (width * height)))
            width = min(max_pixels, max(1, int(width * scale)))
            height = min(max_pixels // width, max(1, int(height * scale)))
            images = convert_from_path(
                str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number,
                size=(width, height),
            )
            try:
                image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            finally:
                for rendered in images:
                    rendered.close()
            yield image


def iter_images(source):
    """Đọc một file ảnh hoặc render lần lượt từng trang PDF."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Không tìm thấy file ảnh/PDF: {path}")
    if path.suffix.lower() == ".pdf":
        for index, image in enumerate(iter_pdf_images(str(path)), 1):
            yield f"{path.stem}_page_{index}", image
        return
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    yield path.stem, image
 