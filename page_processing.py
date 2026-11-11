"""Xử lý một trang: tiền xử lý ảnh, tìm bảng (heron) rồi tìm vùng chữ ngoài bảng (contour)."""

from box_detection import _text_regions
from logger import logger
from ocr_image_preparation import clean_image, prepare_page, resize_large_image
from table_detection import detect_table_regions


def detect_regions(image_bgr, detector, name, *, largest_boxes=None, padding=50):
    """-> (vùng chữ, bảng); mỗi vùng là dict có "box" (xyxy trên ảnh đã xử lý) và "crop".

    Ảnh trang: thu nhỏ ảnh quá lớn, phóng ảnh nhỏ, xoay thẳng, xóa nền, làm nét, cắt viền.
    largest_boxes giới hạn số vùng chữ; bảng luôn được loại khỏi vùng chữ.
    """
    page = clean_image(prepare_page(resize_large_image(image_bgr))[0], padding=padding)
    detected, tables = detect_table_regions(page, detector)
    _, _, texts = _text_regions(page, padding=padding, largest_boxes=largest_boxes,
                                exclude_boxes=[table["box"] for table in tables])
    texts = texts or []
    h, w = image_bgr.shape[:2]
    ph, pw = page.shape[:2]
    logger.info(f"{name}: {w}x{h} → {pw}x{ph}, heron {detected} bảng thô → {len(tables)} bảng, "
                f"{len(texts)} vùng chữ ngoài bảng")
    return texts, tables
