"""PDF/ảnh → phát hiện block chữ và bảng → OCR → trích xuất theo mẫu JSON."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

from PIL import Image

from adaptive_extract import adaptive_extract
import cv2
from extract_info import extract_text, load_templates
from image_io import iter_images
from ocr_image_preparation import enhance_ocr_crop
from page_processing import process_single_image
from qwen_ocr import QwenOCR, TABLE_OCR_PROMPT
from table_detection import load_detector


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output_flow"
OCR_MAX_TIME_SECONDS = 120.0
OCR_MAX_NEW_TOKENS = 2048
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# Số vùng gửi OCR cùng lúc; vLLM tự gom các request đồng thời thành batch.
OCR_WORKERS = 6


@lru_cache(maxsize=1)
def _shared_detector():
    """Nạp heron một lần cho cả tiến trình (API, batch) thay vì mỗi file."""
    return load_detector()


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ocr_image(image, engine, *, prompt=None):
    """Cải thiện ảnh block rồi OCR nguyên ảnh một lần."""
    options = {"max_new_tokens": OCR_MAX_NEW_TOKENS, "max_time": OCR_MAX_TIME_SECONDS}
    if prompt is not None:
        options["prompt"] = prompt
    return engine.ocr_image(enhance_ocr_crop(image), **options)


def _ocr_crop(crop, engine, *, prompt=None):
    """OCR crop trên đĩa và lưu text cạnh file ảnh."""
    with Image.open(crop) as crop_image:
        text = _ocr_image(crop_image, engine, prompt=prompt)
    crop.with_suffix(".ocr.txt").write_text(text, encoding="utf-8")
    return text


def run_flow(source, output=None, template=None, *, engine_factory=None, save_artifacts=True,
             refine_text=True):
    """OCR mọi trang rồi trích xuất toàn tài liệu theo mẫu JSON được chọn."""
    started = time.perf_counter()
    source = Path(source).expanduser().resolve()
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
        print(f"Kết quả: {output}", file=sys.stderr)

    engine_factory = QwenOCR if engine_factory is None else engine_factory
    detector_started = time.perf_counter()
    detector = _shared_detector()
    timings = {"load_detector_seconds": round(time.perf_counter() - detector_started, 3)}
    ocr_engine, cache = None, {}

    def read_pages(largest_boxes):
        nonlocal ocr_engine
        pages = []
        for page, (name, page_image) in enumerate(iter_images(source), 1):
            page_started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                report = process_single_image(page_image, name, output, detector=detector,
                                              save=save_artifacts, include_crops=not save_artifacts,
                                              largest_boxes=largest_boxes, padding=50,
                                              refine_text=refine_text)
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
                crop = output / report["name"] / folder / f"{prefix}_{block['order']:03d}.png"
                cache_key = (page, block_key, tuple(block["box"]))
                if cache_key in cache:
                    if save_artifacts:
                        crop.with_suffix(".ocr.txt").write_text(cache[cache_key], encoding="utf-8")
                    return cache[cache_key]
                print(f"OCR trang {page}, {label} {block['order']}", file=sys.stderr)
                prompt = TABLE_OCR_PROMPT if block_key == "table_blocks" else None
                if save_artifacts:
                    text = _ocr_crop(crop, ocr_engine, prompt=prompt)
                else:
                    with Image.fromarray(cv2.cvtColor(block["crop"], cv2.COLOR_BGR2RGB)) as image:
                        text = _ocr_image(image, ocr_engine, prompt=prompt)
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

    def read_full_pages():
        nonlocal ocr_engine
        pages = []
        for page, (name, page_image) in enumerate(iter_images(source), 1):
            page_started = time.perf_counter()
            if ocr_engine is None:
                ocr_engine = engine_factory()
            print(f"OCR toàn trang {page}", file=sys.stderr)
            with Image.fromarray(cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)) as image:
                text = ocr_engine.ocr_image(
                    image, max_new_tokens=OCR_MAX_NEW_TOKENS, max_time=OCR_MAX_TIME_SECONDS,
                )
            page_output = output / name
            if save_artifacts:
                page_output.mkdir(parents=True, exist_ok=True)
                (page_output / "full_page.ocr.txt").write_text(text, encoding="utf-8")
            pages.append({"page": page, "text": text,
                          "seconds": round(time.perf_counter() - page_started, 3)})
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
    pages, extracted = adaptive_extract(
        lambda limit: save_pages(read_pages(limit), "regions"),
        lambda text: extract_text(text, templates, kind),
        lambda: save_pages(read_full_pages(), "full_page"),
    )
    del ocr_engine
    # Gồm OCR mọi lần đọc lại của adaptive_extract và bước trích xuất.
    timings["ocr_extract_seconds"] = round(time.perf_counter() - extract_started, 3)
    timings["total_seconds"] = round(time.perf_counter() - started, 3)
    print(f"Thời gian: {timings}", file=sys.stderr)

    result = {"source": str(source), "pages": pages, "timings": timings}
    if not any(page["text"].strip() for page in pages):
        raise ValueError("Văn bản OCR rỗng; không thể trích xuất.")
    if not save_artifacts:
        return {**result, "extracted": extracted}
    _write_json(output / "ocr.json", result)
    document_text = "\n\n".join(page["text"] for page in pages) + "\n"
    (output / "ocr.txt").write_text(document_text, encoding="utf-8")
    print(f"Đã lưu OCR: {output / 'ocr.json'} và {output / 'ocr.txt'}", file=sys.stderr)
    _write_json(output / "extracted.json", extracted)
    print(f"Đã lưu trích xuất: {output / 'extracted.json'}", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File PDF hoặc ảnh cần xử lý")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="Thư mục lưu block và OCR (mặc định: output_flow/<tên file>)")
    parser.add_argument("--template", type=Path,
                        help="Chọn mẫu JSON thủ công (mặc định: tự nhận diện từ OCR)")
    parser.add_argument("--no-heron-text", dest="refine_text", action="store_false",
                        help="Không dùng heron tách vùng chữ; OCR trực tiếp box contour (nhanh hơn)")
    args = parser.parse_args()
    try:
        run_flow(args.source, args.output_dir, args.template, refine_text=args.refine_text)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
