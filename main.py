"""PDF/ảnh → phát hiện block chữ và bảng → OCR → trích xuất theo mẫu JSON.

Truyền một thư mục thay vì file để chạy cả loạt và ghi report thời gian theo từng trang."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import shutil
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


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output_flow"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAGE_TIMING_KEYS = ("page", "seconds", "detect_seconds", "ocr_seconds")
OCR_MAX_TIME_SECONDS = 120.0
OCR_MAX_NEW_TOKENS = 2048
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# Số vùng gửi OCR cùng lúc; vLLM tự gom các request đồng thời thành batch.
OCR_WORKERS = 10


@lru_cache(maxsize=1)
def _shared_detector():
    """Nạp heron một lần cho cả tiến trình (API, batch) thay vì mỗi file."""
    return load_detector()


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ocr_image(image, engine, *, prompt=None, raw=False):
    """Cải thiện ảnh block rồi OCR nguyên ảnh một lần; raw=True chỉ giới hạn số pixel."""
    options = {"max_new_tokens": OCR_MAX_NEW_TOKENS, "max_time": OCR_MAX_TIME_SECONDS}
    if prompt is not None:
        options["prompt"] = prompt
    image = limit_ocr_pixels(image) if raw else enhance_ocr_crop(image)
    return engine.ocr_image(image, **options)


def _ocr_crop(crop, engine, *, prompt=None, raw=False):
    """OCR crop trên đĩa và lưu text cạnh file ảnh."""
    with Image.open(crop) as crop_image:
        text = _ocr_image(crop_image, engine, prompt=prompt, raw=raw)
    crop.with_suffix(".ocr.txt").write_text(text, encoding="utf-8")
    return text


def run_flow(source, output=None, template=None, *, engine_factory=None, save_artifacts=True,
             full_page=False, raw=False):
    """OCR mọi trang rồi trích xuất toàn tài liệu theo mẫu JSON được chọn.

    full_page=True: OCR một lần cả trang (đã che bảng), bảng vẫn tìm bằng heron và OCR riêng.
    raw=True: dùng ảnh gốc, bỏ mọi bước xử lý ảnh (chỉ thu nhỏ ảnh cắt về giới hạn của model).
    """
    started = time.perf_counter()
    source = Path(source).expanduser().resolve()
    # idRequest trong log = tên file đang xử lý.
    request_token = request_id_context.set(source.name)
    try:
        return _run_flow(source, output, template, started, engine_factory=engine_factory,
                         save_artifacts=save_artifacts, full_page=full_page, raw=raw)
    except Exception:
        logger.exception("Xử lý thất bại: %s", source)
        raise
    finally:
        request_id_context.reset(request_token)


def _run_flow(source, output, template, started, *, engine_factory, save_artifacts,
              full_page, raw):
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
    output = (Path(output).expanduser().resolve() if output else
              DEFAULT_OUTPUT_DIR / source.stem)
    if save_artifacts:
        output.mkdir(parents=True, exist_ok=True)
        logger.info(f"Thư mục kết quả: {output}")

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
            report = process_single_image(page_image, name, output, detector=detector,
                                          save=save_artifacts, include_crops=not save_artifacts,
                                          largest_boxes=largest_boxes, padding=50,
                                          preprocess=not raw)
            detect_seconds = round(time.perf_counter() - page_started, 3)
            # Chỉ đọc các vùng vừa phát hiện trong report, bỏ qua crop cũ trên đĩa.
            regions = []
            for block_key, folder, prefix, label in (
                ("text_blocks", "crops", "block", "vùng chữ"),
                ("table_blocks", "tables", "table", "bảng"),
            ):
                for block in report.get(block_key, []):
                    regions.append((block, block_key, folder, prefix, label))
            # Dùng box phát hiện, không dùng crop_box có padding của bảng.
            # Cùng độ cao thì đọc từ trái sang phải.
            regions.sort(key=lambda region: (region[0]["box"][1], region[0]["box"][0]))
            if regions and ocr_engine is None:
                ocr_engine = engine_factory()

            def read_region(region):
                block, block_key, folder, prefix, label = region
                request_id_context.set(request_id)  # ContextVar không tự sang thread của pool
                crop = output / report["name"] / folder / f"{prefix}_{block['order']:03d}.png"
                cache_key = (page, block_key, tuple(block["box"]))
                if cache_key in cache:
                    if save_artifacts:
                        crop.with_suffix(".ocr.txt").write_text(cache[cache_key], encoding="utf-8")
                    return cache[cache_key]
                logger.debug(f"OCR trang {page}, {label} {block['order']} box={block['box']}")
                prompt = TABLE_OCR_PROMPT if block_key == "table_blocks" else None
                if save_artifacts:
                    text = _ocr_crop(crop, ocr_engine, prompt=prompt, raw=raw)
                else:
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
            if save_artifacts:
                page_output = output / name
                (page_output / "tables").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(page_output / "full_page.png"), masked)
                (page_output / "full_page.ocr.txt").write_text(texts[0], encoding="utf-8")
                for order, (table, text) in enumerate(zip(tables, texts[1:]), 1):
                    crop = page_output / "tables" / f"table_{order:03d}.png"
                    cv2.imwrite(str(crop), table["crop"])
                    crop.with_suffix(".ocr.txt").write_text(text, encoding="utf-8")
            seconds = round(time.perf_counter() - page_started, 3)
            pages.append({"page": page, "text": "\n\n".join(texts), "seconds": seconds,
                          "detect_seconds": detect_seconds,
                          "ocr_seconds": round(seconds - detect_seconds, 3)})
        return pages

    def save_pages(pages, prefix):
        if not pages:
            raise ValueError("Đầu vào không có ảnh/trang để xử lý.")
        if not save_artifacts:
            return pages
        _write_json(output / f"{prefix}.ocr.json", {"source": str(source), "pages": pages})
        text = "\n\n".join(page["text"] for page in pages) + "\n"
        (output / f"{prefix}.ocr.txt").write_text(text, encoding="utf-8")
        return pages

    extract_started = time.perf_counter()
    if full_page:
        pages = save_pages(read_full_pages_with_tables(), "full_page")
        extracted = extract_text("\n\n".join(page["text"] for page in pages) + "\n", templates, kind)
    else:
        pages, extracted = adaptive_extract(
            lambda limit: save_pages(read_pages(limit), "regions"),
            lambda text: extract_text(text, templates, kind),
            lambda: save_pages(read_full_pages_with_tables(), "full_page"),
        )
    del ocr_engine
    # Gồm OCR mọi lần đọc lại của adaptive_extract và bước trích xuất.
    timings["ocr_extract_seconds"] = round(time.perf_counter() - extract_started, 3)
    timings["total_seconds"] = round(time.perf_counter() - started, 3)
    logger.info(f"Thời gian: {timings}")

    result = {"source": str(source), "pages": pages, "timings": timings}
    if not any(page["text"].strip() for page in pages):
        raise ValueError("Văn bản OCR rỗng; không thể trích xuất.")
    if not save_artifacts:
        return {**result, "extracted": extracted}
    _write_json(output / "ocr.json", result)
    document_text = "\n\n".join(page["text"] for page in pages) + "\n"
    (output / "ocr.txt").write_text(document_text, encoding="utf-8")
    logger.info(f"Đã lưu OCR: {output / 'ocr.json'} và {output / 'ocr.txt'}")
    _write_json(output / "extracted.json", extracted)
    logger.info(f"Đã lưu trích xuất: {output / 'extracted.json'}")
    return result


def run_batch(data_dir, output_dir, template=None, full_page=False, raw=False):
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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

    rows, documents, used_names = [], [], set()
    batch_started = time.perf_counter()
    for index, source in enumerate(files, 1):
        relative = source.relative_to(data_dir)
        logger.info(f"[{index}/{len(files)}] {relative}")
        started = time.perf_counter()
        doc = {"file": str(relative), "status": "ok", "error": None, "pages": []}
        # Mỗi file một thư mục <tên file> ngay dưới output_dir (không giữ thư mục con của dữ liệu),
        # chứa bản sao file gốc và extract.json; trùng tên thì thêm _2, _3...
        name, suffix = source.stem, 2
        while name in used_names:
            name, suffix = f"{source.stem}_{suffix}", suffix + 1
        used_names.add(name)
        doc_dir = output_dir / name
        try:
            result = run_flow(source, doc_dir, template, engine_factory=engine_factory,
                              save_artifacts=False, full_page=full_page, raw=raw)
            doc_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, doc_dir / source.name)
            _write_json(doc_dir / "extract.json", result["extracted"])
            doc["pages"] = [{key: p.get(key) for key in PAGE_TIMING_KEYS} for p in result["pages"]]
            doc["timings"] = result.get("timings", {})
        except Exception as exc:  # noqa: BLE001 - ghi lỗi rồi chạy tiếp file khác
            doc.update(status="error", error=f"{type(exc).__name__}: {exc}")
            logger.error(f"[{index}/{len(files)}] {relative} lỗi: {doc['error']}")
        doc["total_seconds"] = round(time.perf_counter() - started, 3)
        documents.append(doc)
        for p in doc["pages"] or [dict.fromkeys(PAGE_TIMING_KEYS)]:
            rows.append({"file": doc["file"], "page": p["page"], "page_seconds": p["seconds"],
                         "detect_seconds": p["detect_seconds"], "ocr_seconds": p["ocr_seconds"],
                         "file_total_seconds": doc["total_seconds"], "status": doc["status"],
                         "error": doc["error"]})
        logger.info(f"[{index}/{len(files)}] {relative}: {doc['total_seconds']} giây, {len(doc['pages'])} trang")
        # Ghi report sau mỗi file để không mất dữ liệu nếu bị dừng giữa chừng.
        summary = _write_report(output_dir, rows, documents,
                                round(time.perf_counter() - batch_started, 3))

    if documents:
        logger.info(f"Tổng kết: {summary['files']} file ({summary['failed_files']} lỗi), "
                    f"{summary['pages']} trang trong {summary['wall_seconds']} giây "
                    f"({summary['pages_per_minute']} trang/phút)")
        for prefix, label in (("file", "Mỗi file"), ("page", "Mỗi trang"),
                              ("detect", "  phát hiện"), ("ocr", "  OCR")):
            logger.info(f"  {label}: TB {summary[f'avg_{prefix}_seconds']} giây, "
                        f"min {summary[f'min_{prefix}_seconds']}, max {summary[f'max_{prefix}_seconds']}")
    logger.info(f"Report: {output_dir / 'timing_report.csv'}, {output_dir / 'timing_report.json'}")
    return documents


def _stats(values, prefix):
    values = [v for v in values if v is not None]
    return {
        f"total_{prefix}_seconds": round(sum(values), 3),
        f"avg_{prefix}_seconds": round(sum(values) / len(values), 3) if values else None,
        f"max_{prefix}_seconds": max(values, default=None),
        f"min_{prefix}_seconds": min(values, default=None),
    }


def _write_report(output_dir, rows, documents, wall_seconds):
    with open(output_dir / "timing_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ok = [d for d in documents if d["status"] == "ok"]
    pages = [p for d in ok for p in d["pages"]]
    summary = {
        "files": len(documents),
        "failed_files": len(documents) - len(ok),
        "pages": len(pages),
        "wall_seconds": wall_seconds,
        **_stats([d["total_seconds"] for d in ok], "file"),
        **_stats([p["seconds"] for p in pages], "page"),
        **_stats([p["detect_seconds"] for p in pages], "detect"),
        **_stats([p["ocr_seconds"] for p in pages], "ocr"),
        "load_detector_seconds": round(sum(d.get("timings", {}).get("load_detector_seconds", 0)
                                           for d in ok), 3),
        "pages_per_minute": round(60 * len(pages) / wall_seconds, 2) if wall_seconds else None,
    }
    _write_json(output_dir / "timing_report.json", {"summary": summary, "documents": documents})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_DATA_DIR,
                        help="File PDF/ảnh, hoặc thư mục để chạy batch (mặc định: ../data)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="Thư mục kết quả (mặc định: output_flow/<tên file>, "
                             "batch: output_flow/batch)")
    parser.add_argument("--template", type=Path,
                        help="Chọn mẫu JSON thủ công (mặc định: tự nhận diện từ OCR)")
    parser.add_argument("--full-page", action="store_true",
                        help="OCR một lần cả trang (bỏ contour), bảng vẫn OCR riêng")
    parser.add_argument("--raw", action="store_true",
                        help="Dùng ảnh gốc, không xử lý ảnh (xoay thẳng, xóa nền, làm nét...)")
    args = parser.parse_args()
    if args.source.is_dir():
        run_batch(args.source, args.output_dir or DEFAULT_OUTPUT_DIR / "batch",
                  args.template, args.full_page, args.raw)
        return 0
    try:
        run_flow(args.source, args.output_dir, args.template, full_page=args.full_page,
                 raw=args.raw)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error(f"Lỗi: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
