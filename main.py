"""PDF/ảnh → phát hiện vùng chữ và bảng → OCR → trích xuất theo mẫu JSON.

Kết quả trích xuất ghi vào log, không lưu file. Truyền một thư mục để chạy cả loạt."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
from PIL import Image

from adaptive_extract import adaptive_extract
from extract_info import extract_text, load_templates
from image_io import iter_images
from logger import logger, request_id_context
from ocr_image_preparation import enhance_ocr_crop
from page_processing import detect_regions
from qwen_ocr import QwenOCR, TABLE_OCR_PROMPT
from table_detection import detect_table_regions, load_detector


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
OCR_OPTIONS = {"max_new_tokens": 2048, "max_time": 120.0}
# Số vùng gửi OCR cùng lúc; vLLM tự gom các request đồng thời thành batch.
OCR_WORKERS = 10


@lru_cache(maxsize=1)
def _detector():
    """Nạp heron một lần cho cả tiến trình."""
    return load_detector()


def _to_pil(image_bgr):
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _ocr_all(engine, jobs):
    """jobs: [(ảnh PIL, prompt hoặc None)] -> text theo đúng thứ tự, OCR song song."""
    request_id = request_id_context.get()

    def ocr(job):
        image, prompt = job
        request_id_context.set(request_id)  # ContextVar không tự sang thread của pool
        options = OCR_OPTIONS if prompt is None else {**OCR_OPTIONS, "prompt": prompt}
        return engine.ocr_image(image, **options)

    with ThreadPoolExecutor(OCR_WORKERS) as pool:
        return list(pool.map(ocr, jobs))


def _read_regions(source, engine, largest_boxes):
    """Mỗi trang: OCR từng vùng chữ (contour) và từng bảng, nối theo thứ tự đọc."""
    pages = []
    for page, (name, image) in enumerate(iter_images(source), 1):
        text_blocks, table_blocks = detect_regions(image, _detector(), name,
                                                   largest_boxes=largest_boxes)
        blocks = [(b, None) for b in text_blocks] + [(b, TABLE_OCR_PROMPT) for b in table_blocks]
        # Cùng độ cao thì đọc từ trái sang phải.
        blocks.sort(key=lambda item: (item[0]["box"][1], item[0]["box"][0]))
        texts = _ocr_all(engine, [(enhance_ocr_crop(_to_pil(b["crop"])), prompt)
                                  for b, prompt in blocks])
        pages.append({"page": page, "text": "\n\n".join(texts)})
    return pages


def _read_full_pages(source, engine):
    """Mỗi trang: OCR một lần cả trang (đã che bảng, ảnh gốc), bảng OCR riêng nối phía sau."""
    pages = []
    for page, (_, image) in enumerate(iter_images(source), 1):
        _, tables = detect_table_regions(image, _detector())
        masked = image.copy()
        for x1, y1, x2, y2 in (table["box"] for table in tables):
            masked[y1:y2, x1:x2] = 255
        logger.info(f"OCR toàn trang {page} (che {len(tables)} bảng)")
        jobs = [(_to_pil(masked), None)] + [(enhance_ocr_crop(_to_pil(t["crop"])), TABLE_OCR_PROMPT)
                                            for t in tables]
        pages.append({"page": page, "text": "\n\n".join(_ocr_all(engine, jobs))})
    return pages


def run_flow(source, engine=None, templates=None):
    """OCR rồi trích xuất một file; kết quả ghi vào log và trả về dict."""
    source = Path(source).expanduser().resolve()
    request_token = request_id_context.set(source.name)  # idRequest trong log = tên file
    started = time.perf_counter()
    try:
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Đầu vào phải là file PDF hoặc ảnh.")
        logger.info(f"Bắt đầu xử lý {source}")
        engine = engine or QwenOCR()
        templates = templates or load_templates()

        def checked(pages):
            if not pages:
                raise ValueError("Đầu vào không có ảnh/trang để xử lý.")
            return pages

        pages, extracted = adaptive_extract(
            lambda limit: checked(_read_regions(source, engine, limit)),
            lambda text: extract_text(text, templates),
            lambda: checked(_read_full_pages(source, engine)),
        )
        if not any(page["text"].strip() for page in pages):
            raise ValueError("Văn bản OCR rỗng; không thể trích xuất.")
        logger.info(f"Xong {len(pages)} trang trong {time.perf_counter() - started:.3f} giây")
        logger.info("Kết quả trích xuất: %s", json.dumps(extracted, ensure_ascii=False))
        return {"source": str(source), "pages": pages, "extracted": extracted}
    except Exception:
        logger.exception(f"Xử lý thất bại: {source}")
        raise
    finally:
        request_id_context.reset(request_token)


def run_batch(data_dir):
    """Chạy mọi file PDF/ảnh trong thư mục (kể cả thư mục con); trả về số file lỗi."""
    data_dir = Path(data_dir).expanduser().resolve()
    files = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    logger.info(f"Tìm thấy {len(files)} file trong {data_dir}")
    engine, templates = QwenOCR(), load_templates()
    failed, pages, started = 0, 0, time.perf_counter()
    for index, source in enumerate(files, 1):
        logger.info(f"[{index}/{len(files)}] {source.relative_to(data_dir)}")
        try:
            pages += len(run_flow(source, engine, templates)["pages"])
        except Exception:  # noqa: BLE001 - đã ghi log trong run_flow, chạy tiếp file khác
            failed += 1
    logger.info(f"Tổng kết: {len(files)} file ({failed} lỗi), {pages} trang trong "
                f"{time.perf_counter() - started:.3f} giây")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_DATA_DIR,
                        help="File PDF/ảnh, hoặc thư mục để chạy batch (mặc định: ../data)")
    source = parser.parse_args().source
    if source.is_dir():
        return 1 if run_batch(source) else 0
    try:
        run_flow(source)
    except Exception:  # noqa: BLE001 - lỗi đã ghi log trong run_flow
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
