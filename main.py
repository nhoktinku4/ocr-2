"""PDF/ảnh → phát hiện block chữ và bảng → OCR → trích xuất theo mẫu JSON.

Kết quả trích xuất ghi vào log, không lưu file. Truyền một thư mục để chạy cả loạt."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import time

from PIL import Image

from adaptive_extract import adaptive_extract
import cv2
from extract_info import extract_text, load_templates
from image_io import iter_images
from logger import logger, request_id_context
from ocr_image_preparation import enhance_ocr_crop, limit_ocr_pixels
from page_processing import process_single_image
from qwen_ocr import QwenOCR, TABLE_OCR_PROMPT
from table_detection import detect_table_regions, load_detector


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OCR_MAX_TIME_SECONDS = 120.0
OCR_MAX_NEW_TOKENS = 2048
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# Số vùng gửi OCR cùng lúc; vLLM tự gom các request đồng thời thành batch.
OCR_WORKERS = 10


@lru_cache(maxsize=1)
def _shared_detector():
    """Nạp heron một lần cho cả tiến trình (API, batch) thay vì mỗi file."""
    return load_detector()


def _ocr_image(image, engine, *, prompt=None, raw=False):
    """Cải thiện ảnh block rồi OCR nguyên ảnh một lần; raw=True chỉ giới hạn số pixel."""
    options = {"max_new_tokens": OCR_MAX_NEW_TOKENS, "max_time": OCR_MAX_TIME_SECONDS}
    if prompt is not None:
        options["prompt"] = prompt
    image = limit_ocr_pixels(image) if raw else enhance_ocr_crop(image)
    return engine.ocr_image(image, **options)


def run_flow(source, template=None, *, engine_factory=None,
             full_page=False, raw=False):
    """OCR mọi trang rồi trích xuất toàn tài liệu theo mẫu JSON; kết quả ghi vào log.

    full_page=True: OCR một lần cả trang (đã che bảng), bảng vẫn tìm bằng heron và OCR riêng.
    raw=True: dùng ảnh gốc, bỏ mọi bước xử lý ảnh (chỉ thu nhỏ ảnh cắt về giới hạn của model).
    """
    started = time.perf_counter()
    source = Path(source).expanduser().resolve()
    # idRequest trong log = tên file đang xử lý.
    request_token = request_id_context.set(source.name)
    try:
        return _run_flow(source, template, started, engine_factory=engine_factory,
                         full_page=full_page, raw=raw)
    except Exception:
        logger.exception("Xử lý thất bại: %s", source)
        raise
    finally:
        request_id_context.reset(request_token)


def _run_flow(source, template, started, *, engine_factory, full_page, raw):
    logger.info("Bắt đầu xử lý %s (full_page=%s, raw=%s)", source, full_page, raw)
    if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Đầu vào phải là file PDF hoặc ảnh.")
    templates, kind = load_templates(), None
    if template is not None:
        # Mẫu chọn thủ công quyết định luôn loại giấy, không tự nhận diện.
        template_path = Path(template).expanduser().resolve()
        extraction_template = json.loads(template_path.read_text(encoding="utf-8"))
        if not isinstance(extraction_template, dict):
            raise ValueError("Mẫu JSON phải là object.")
        kind = "hkd" if "ma_so_ho_kinh_doanh" in extraction_template else "dn"
        templates[kind] = extraction_template

    engine_factory = QwenOCR if engine_factory is None else engine_factory
    detector_started = time.perf_counter()
    detector = _shared_detector()
    timings = {"load_detector_seconds": round(time.perf_counter() - detector_started, 3)}
    request_id = request_id_context.get()
    ocr_engine, cache = None, {}

    def read_pages(largest_boxes):
        nonlocal ocr_engine
        pages = []
        for page, (name, page_image) in enumerate(iter_images(source), 1):
            page_started = time.perf_counter()
            report = process_single_image(page_image, name, Path("."), detector=detector,
                                          save=False, include_crops=True,
                                          largest_boxes=largest_boxes, padding=50,
                                          preprocess=not raw)
            detect_seconds = round(time.perf_counter() - page_started, 3)
            # Chỉ đọc các vùng vừa phát hiện trong report, bỏ qua crop cũ trên đĩa.
            regions = []
            for block_key, label in (("text_blocks", "vùng chữ"), ("table_blocks", "bảng")):
                for block in report.get(block_key, []):
                    regions.append((block, block_key, label))
            # Dùng box phát hiện, không dùng crop_box có padding của bảng.
            # Cùng độ cao thì đọc từ trái sang phải.
            regions.sort(key=lambda region: (region[0]["box"][1], region[0]["box"][0]))
            if regions and ocr_engine is None:
                ocr_engine = engine_factory()

            def read_region(region):
                block, block_key, label = region
                request_id_context.set(request_id)  # ContextVar không tự sang thread của pool
                cache_key = (page, block_key, tuple(block["box"]))
                if cache_key in cache:
                    return cache[cache_key]
                logger.debug(f"OCR trang {page}, {label} {block['order']} box={block['box']}")
                prompt = TABLE_OCR_PROMPT if block_key == "table_blocks" else None
                with Image.fromarray(cv2.cvtColor(block["crop"], cv2.COLOR_BGR2RGB)) as image:
                    text = _ocr_image(image, ocr_engine, prompt=prompt, raw=raw)
                cache[cache_key] = text
                return text

            # map giữ đúng thứ tự đọc dù các vùng xong không theo thứ tự.
            with ThreadPoolExecutor(OCR_WORKERS) as pool:
                texts = list(pool.map(read_region, regions))
            seconds = round(time.perf_counter() - page_started, 3)
            pages.append({"page": page, "text": "\n\n".join(texts), "seconds": seconds,
                          "detect_seconds": detect_seconds,
                          "ocr_seconds": round(seconds - detect_seconds, 3)})
        return pages

    def read_full_pages_with_tables():
        """Bảng (heron) OCR riêng bằng prompt bảng; phần còn lại của trang OCR một lần.

        Trang dùng thẳng ảnh gốc, không tiền xử lý; bảng cũng tìm trên ảnh gốc để box che khớp.
        """
        nonlocal ocr_engine
        pages = []
        for page, (name, page_image) in enumerate(iter_images(source), 1):
            page_started = time.perf_counter()
            _, tables = detect_table_regions(page_image, detector)
            masked = page_image.copy()
            for x1, y1, x2, y2 in (table["box"] for table in tables):
                masked[y1:y2, x1:x2] = 255
            detect_seconds = round(time.perf_counter() - page_started, 3)
            if ocr_engine is None:
                ocr_engine = engine_factory()
            logger.info(f"OCR toàn trang {page} (che {len(tables)} bảng)")

            def read_region(region):
                image, prompt = region
                request_id_context.set(request_id)
                with Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) as pil:
                    if prompt is None:
                        return ocr_engine.ocr_image(pil, max_new_tokens=OCR_MAX_NEW_TOKENS,
                                                    max_time=OCR_MAX_TIME_SECONDS)
                    return _ocr_image(pil, ocr_engine, prompt=prompt, raw=raw)

            # Bảng xếp theo thứ tự từ trên xuống, nối sau phần chữ của trang.
            tables.sort(key=lambda table: (table["box"][1], table["box"][0]))
            regions = [(masked, None)] + [(table["crop"], TABLE_OCR_PROMPT) for table in tables]
            with ThreadPoolExecutor(OCR_WORKERS) as pool:
                texts = list(pool.map(read_region, regions))
            seconds = round(time.perf_counter() - page_started, 3)
            pages.append({"page": page, "text": "\n\n".join(texts), "seconds": seconds,
                          "detect_seconds": detect_seconds,
                          "ocr_seconds": round(seconds - detect_seconds, 3)})
        return pages

    def check_pages(pages):
        if not pages:
            raise ValueError("Đầu vào không có ảnh/trang để xử lý.")
        return pages

    extract_started = time.perf_counter()
    if full_page:
        pages = check_pages(read_full_pages_with_tables())
        extracted = extract_text("\n\n".join(page["text"] for page in pages) + "\n", templates, kind)
    else:
        pages, extracted = adaptive_extract(
            lambda limit: check_pages(read_pages(limit)),
            lambda text: extract_text(text, templates, kind),
            lambda: check_pages(read_full_pages_with_tables()),
        )
    del ocr_engine
    # Gồm OCR mọi lần đọc lại của adaptive_extract và bước trích xuất.
    timings["ocr_extract_seconds"] = round(time.perf_counter() - extract_started, 3)
    timings["total_seconds"] = round(time.perf_counter() - started, 3)
    logger.info(f"Thời gian: {timings}")

    result = {"source": str(source), "pages": pages, "timings": timings}
    if not any(page["text"].strip() for page in pages):
        raise ValueError("Văn bản OCR rỗng; không thể trích xuất.")
    logger.info("Kết quả trích xuất: %s", json.dumps(extracted, ensure_ascii=False))
    return {**result, "extracted": extracted}


def run_batch(data_dir, template=None, full_page=False, raw=False):
    data_dir = Path(data_dir).expanduser().resolve()
    files = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    logger.info(f"Tìm thấy {len(files)} file trong {data_dir}")

    # Nạp model OCR một lần, dùng chung cho mọi file.
    engine = None

    def engine_factory():
        nonlocal engine
        if engine is None:
            engine = QwenOCR()
        return engine

    failed, pages = 0, 0
    batch_started = time.perf_counter()
    for index, source in enumerate(files, 1):
        relative = source.relative_to(data_dir)
        logger.info(f"[{index}/{len(files)}] {relative}")
        try:
            result = run_flow(source, template, engine_factory=engine_factory,
                              full_page=full_page, raw=raw)
            pages += len(result["pages"])
            logger.info(f"[{index}/{len(files)}] {relative}: {result['timings']['total_seconds']} giây, "
                        f"{len(result['pages'])} trang")
        except Exception as exc:  # noqa: BLE001 - ghi lỗi rồi chạy tiếp file khác
            failed += 1
            logger.error(f"[{index}/{len(files)}] {relative} lỗi: {type(exc).__name__}: {exc}")
    logger.info(f"Tổng kết: {len(files)} file ({failed} lỗi), {pages} trang trong "
                f"{round(time.perf_counter() - batch_started, 3)} giây")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_DATA_DIR,
                        help="File PDF/ảnh, hoặc thư mục để chạy batch (mặc định: ../data)")
    parser.add_argument("--template", type=Path,
                        help="Chọn mẫu JSON thủ công (mặc định: tự nhận diện từ OCR)")
    parser.add_argument("--full-page", action="store_true",
                        help="OCR một lần cả trang (bỏ contour), bảng vẫn OCR riêng")
    parser.add_argument("--raw", action="store_true",
                        help="Dùng ảnh gốc, không xử lý ảnh (xoay thẳng, xóa nền, làm nét...)")
    args = parser.parse_args()
    if args.source.is_dir():
        failed = run_batch(args.source, args.template, args.full_page, args.raw)
        return 1 if failed else 0
    try:
        run_flow(args.source, args.template, full_page=args.full_page,
                 raw=args.raw)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error(f"Lỗi: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
