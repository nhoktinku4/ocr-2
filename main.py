"""PDF/ảnh GPKD (doanh nghiệp / hộ kinh doanh) → Qwen-VL đọc cả trang, tự nhận loại → JSON theo json/*.json.

Mọi trang của một giấy được gửi trong một request để model trả về một JSON; nếu request đó lỗi
(context/KV cache vLLM không đủ) thì OCR từng trang rồi gộp. Kết quả lưu vào <output>/<tên file>/:
extract.json, response.txt (câu trả lời thô của model). Truyền một thư mục để chạy cả loạt;
truyền nhiều ảnh để gộp chúng thành một giấy.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import unicodedata

import cv2
from PIL import Image

from image_io import iter_images
from logger import logger, request_id_context
import re

from qwen_ocr import BUSINESS_LINES_KEY, JSON_DIR, SIGNER_KEY, OCR_PROMPT, TYPE_DESCRIPTIONS, QwenOCR


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output_flow"
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
OCR_OPTIONS = {"max_new_tokens": 3000, "max_time": 300.0}
# Số trang OCR song song khi phải chạy từng trang; nên bằng --max-num-seqs của vLLM.
OCR_WORKERS = 5
# Phải bằng --max-model-len của vLLM. Ảnh + prompt + output phải nằm gọn trong context này.
MAX_MODEL_LEN = int(os.getenv("QWEN_OCR_MAX_MODEL_LEN", "16384"))
PIXELS_PER_TOKEN = 28 * 28  # Qwen-VL: mỗi token ảnh ~ một ô 28x28 pixel
MIN_PAGE_PIXELS = 400_000   # dưới mức này chữ quá nhỏ, OCR không còn tin cậy


def load_templates():
    """json/<loại>.json → {loại: dict mẫu}."""
    templates = {}
    for kind in TYPE_DESCRIPTIONS:
        with open(os.path.join(JSON_DIR, f"{kind}.json"), encoding="utf-8") as f:
            templates[kind] = json.load(f)
    return templates


def _to_pil(image_bgr):
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _plain(text):
    """Bỏ dấu, chữ thường: dùng để so khớp loại giấy."""
    text = unicodedata.normalize("NFD", str(text)).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _parse_json(text):
    """Lấy object JSON từ câu trả lời của model (bỏ ```json nếu có)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model không trả về JSON: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _empty(value):
    return value is None or value == "" or (isinstance(value, str) and value.strip().lower() == "null")


def _merge_value(template, values):
    """Gộp các giá trị cùng một khóa của nhiều trang theo cấu trúc mẫu."""
    if isinstance(template, dict):
        dicts = [v for v in values if isinstance(v, dict)]
        return {k: _merge_value(t, [d.get(k) for d in dicts]) for k, t in template.items()}
    if isinstance(template, list):
        merged, seen = [], set()
        item_tpl = template[0] if template else None
        for value in values:
            for item in value if isinstance(value, list) else []:
                if isinstance(item_tpl, dict) and isinstance(item, dict):
                    item = _merge_value(item_tpl, [item])
                    if all(_empty(v) for v in item.values()):
                        continue
                    key = (item.get("ho_ten") or json.dumps(item, ensure_ascii=False)).strip().lower()
                elif _empty(item):
                    continue
                else:
                    key = json.dumps(item, ensure_ascii=False).strip().lower()
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        return merged
    return next((v for v in values if not _empty(v)), None)


def _kind_of(result, templates):
    """Loại giấy của một kết quả: theo loai_giay_to, không khớp thì theo số khóa trùng với mẫu."""
    label = _plain(result.get("loai_giay_to") or "")
    if "ho kinh doanh" in label:
        return "ho_kinh_doanh"
    if "doanh nghiep" in label:
        return "doanh_nghiep"
    return max(templates, key=lambda kind: len(set(result) & set(templates[kind])))


def _main_business_line(result):
    """danh_sach_nganh_nghe → nganh_nghe_chinh: dòng có chữ "chính" ở mã ngành; không có thì lấy hết."""
    rows = result.pop(BUSINESS_LINES_KEY, None)
    if not isinstance(rows, list):
        return result
    rows = [r for r in rows if isinstance(r, dict) and not (_empty(r.get("ten")) and _empty(r.get("ma_nganh")))]
    if not rows:
        return result
    main = [r for r in rows if "chinh" in _plain(r.get("ma_nganh") or "")]
    # Model đôi khi để chữ "(chính)" sang cột tên ngành.
    main = main or [r for r in rows if re.search(r"\(\s*chinh\s*\)", _plain(r.get("ten") or ""))]
    chosen = main[:1] or rows

    def clean(value, code):
        value = str(value or "").strip()
        if code:
            return re.sub(r"\D+", "", value) or value
        return re.sub(r"\s*\(\s*ch[ií]nh\s*\)\s*$", "", value, flags=re.I).strip()

    names = [clean(r.get("ten"), False) for r in chosen]
    codes = [clean(r.get("ma_nganh"), True) for r in chosen]
    result["nganh_nghe_chinh"] = {"ten": "; ".join(n for n in names if n) or None,
                                  "ma_nganh": "; ".join(c for c in codes if c) or None}
    return result


PERSON_LIST_KEYS = ("chu_so_huu", "nguoi_dai_dien")


def _name_key(name):
    return " ".join(_plain(name or "").split())


def _drop_signers(result):
    """Bỏ khỏi danh sách người: người ký (theo nguoi_ky) và người chỉ có tên, không có chi tiết nào khác
    (tên đứng riêng ở khối chữ ký, không phải chủ sở hữu / người đại diện)."""
    signer = _name_key(result.pop(SIGNER_KEY, None))
    for key in PERSON_LIST_KEYS:
        people = result.get(key)
        if not isinstance(people, list):
            continue
        kept = []
        for person in people:
            if not isinstance(person, dict):
                continue
            name = _name_key(person.get("ho_ten"))
            details = [v for k, v in person.items() if k not in ("ho_ten", "chuc_danh") and not _empty(v)]
            if (signer and name == signer) or (name and not details):
                logger.info(f"Bỏ {person.get('ho_ten')!r} khỏi {key} (người ký / không có thông tin khác)")
                continue
            kept.append(person)
        result[key] = kept
    if signer and isinstance(result.get("chu_ho"), dict) and _name_key(result["chu_ho"].get("ho_ten")) == signer:
        logger.info(f"chu_ho trùng tên người ký {result['chu_ho'].get('ho_ten')!r}; bỏ tên")
        result["chu_ho"] = {**result["chu_ho"], "ho_ten": None}
    return result


_PROVINCE_RE = re.compile(r"\s+(t[iỉ]nh|th[aà]nh\s+ph[oố]|tp\.?)\s+(.+)$", re.IGNORECASE)


def _split_province(result):
    """co_quan_cap.so dính tên tỉnh ("SỞ KẾ HOẠCH VÀ ĐẦU TƯ TỈNH BÌNH PHƯỚC") → tách sang tinh_thanh."""
    agency = result.get("co_quan_cap")
    if not isinstance(agency, dict) or _empty(agency.get("so")):
        return result
    match = _PROVINCE_RE.search(str(agency["so"]).strip())
    if match:
        agency = {**agency, "so": agency["so"].strip()[:match.start()].strip()}
        if _empty(agency.get("tinh_thanh")):
            agency["tinh_thanh"] = match.group(2).strip()
        result["co_quan_cap"] = agency
    return result


def _merge_pages(results, templates):
    """Gộp JSON của các trang thành một kết quả theo mẫu của loại xuất hiện nhiều nhất."""
    results = [_split_province(_drop_signers(_main_business_line(dict(r)))) for r in results]
    kinds = [_kind_of(r, templates) for r in results]
    kind = max(set(kinds), key=kinds.count)
    merged = _merge_value(templates[kind], [r for r, k in zip(results, kinds) if k == kind])
    merged["loai_giay_to"] = templates[kind]["loai_giay_to"]
    return merged


def _pixel_budget(pages):
    """Tổng pixel cho mọi trang sao cho không vượt context; báo lỗi rõ nếu context quá nhỏ."""
    prompt_tokens = len(OCR_PROMPT) // 2  # ước lượng dư (tiếng Việt + JSON)
    image_tokens = MAX_MODEL_LEN - prompt_tokens - OCR_OPTIONS["max_new_tokens"] - 64 * pages - 256
    total = image_tokens * PIXELS_PER_TOKEN
    if total < MIN_PAGE_PIXELS * pages:
        raise ValueError(
            f"Context {MAX_MODEL_LEN} token không đủ cho {pages} trang (còn ~{max(image_tokens, 0)} token cho ảnh). "
            f"Tăng --max-model-len của vLLM (và QWEN_OCR_MAX_MODEL_LEN) hoặc giảm max_new_tokens.")
    return total


def _ocr_pages_separately(engine, images, templates):
    """Phương án dự phòng: mỗi trang một request (song song), rồi gộp JSON các trang."""
    request_id = request_id_context.get()

    def ocr(image):
        request_id_context.set(request_id)  # ContextVar không tự sang thread của pool
        return engine.ocr_images([image], OCR_PROMPT, **OCR_OPTIONS)

    with ThreadPoolExecutor(OCR_WORKERS) as pool:
        texts = list(pool.map(ocr, images))
    results = []
    for page, text in enumerate(texts, 1):
        try:
            results.append(_parse_json(text))
        except ValueError as exc:
            logger.warning(f"Trang {page}: bỏ qua, JSON lỗi ({exc})")
    if not results:
        raise ValueError("Không trang nào trả về JSON hợp lệ.")
    return "\n\n".join(f"===== Trang {i} =====\n{t}" for i, t in enumerate(texts, 1)), _merge_pages(results, templates)


def extract(images, engine, templates):
    """Ảnh PIL các trang của MỘT giấy → (câu trả lời thô, dict JSON)."""
    if len(images) == 1:
        text = engine.ocr_images(images, OCR_PROMPT, **OCR_OPTIONS)
        return text, _merge_pages([_parse_json(text)], templates)
    logger.info(f"OCR {len(images)} trang trong một request")
    try:
        text = engine.ocr_images(images, OCR_PROMPT, max_total_pixels=_pixel_budget(len(images)), **OCR_OPTIONS)
        return text, _merge_pages([_parse_json(text)], templates)
    except (RuntimeError, ValueError) as exc:
        logger.warning(f"Request nhiều trang thất bại ({exc}); chuyển sang OCR từng trang rồi gộp")
    return _ocr_pages_separately(engine, images, templates)


def run_flow(source, output=None, engine=None, templates=None):
    """OCR một giấy (một file, hoặc danh sách file ảnh là các trang theo thứ tự); lưu vào output
    (mặc định output_flow/<tên file đầu tiên>) và trả về dict."""
    sources = [Path(p).expanduser().resolve() for p in (source if isinstance(source, (list, tuple)) else [source])]
    request_token = request_id_context.set(sources[0].name)  # idRequest trong log = tên file
    started = time.perf_counter()
    try:
        for path in sources:
            if not path.is_file():
                # Trong Docker: chỉ thấy file nằm trong thư mục đã mount.
                raise FileNotFoundError(f"Không tìm thấy file: {path}")
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Đầu vào phải là file PDF hoặc ảnh: {path}")
        output = Path(output) if output else DEFAULT_OUTPUT_DIR / sources[0].stem
        output.mkdir(parents=True, exist_ok=True)
        logger.info(f"Bắt đầu xử lý {', '.join(map(str, sources))} → {output}")
        images = [_to_pil(image) for path in sources for _, image in iter_images(path)]
        if not images:
            raise ValueError("Đầu vào không có ảnh/trang để xử lý.")
        text, extracted = extract(images, engine or QwenOCR(), templates or load_templates())
        (output / "response.txt").write_text(text + "\n", encoding="utf-8")
        (output / "extract.json").write_text(json.dumps(extracted, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        logger.info(f"Xong {len(images)} trang trong {time.perf_counter() - started:.3f} giây")
        logger.info("Kết quả trích xuất: %s", json.dumps(extracted, ensure_ascii=False))
        return {"source": ", ".join(map(str, sources)), "pages": len(images), "text": text,
                "extracted": extracted}
    except Exception:
        logger.exception(f"Xử lý thất bại: {', '.join(map(str, sources))}")
        raise
    finally:
        request_id_context.reset(request_token)


def run_batch(data_dir, output_dir=DEFAULT_OUTPUT_DIR):
    """Chạy mọi file PDF/ảnh trong thư mục (kể cả thư mục con); trả về số file lỗi."""
    data_dir = Path(data_dir).expanduser().resolve()
    files = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    logger.info(f"Tìm thấy {len(files)} file trong {data_dir}")
    engine, templates = QwenOCR(), load_templates()
    failed, pages, started, used = 0, 0, time.perf_counter(), set()
    for index, source in enumerate(files, 1):
        logger.info(f"[{index}/{len(files)}] {source.relative_to(data_dir)}")
        name, n = source.stem, 2                # trùng tên file ở thư mục con khác: thêm _2, _3...
        while name in used:
            name, n = f"{source.stem}_{n}", n + 1
        used.add(name)
        try:
            pages += run_flow(source, Path(output_dir) / name, engine, templates)["pages"]
        except Exception:  # noqa: BLE001 - đã ghi log trong run_flow, chạy tiếp file khác
            failed += 1
    logger.info(f"Tổng kết: {len(files)} file ({failed} lỗi), {pages} trang trong "
                f"{time.perf_counter() - started:.3f} giây")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, nargs="*", default=[DEFAULT_DATA_DIR],
                        help="File PDF/ảnh, thư mục để chạy batch (mặc định: ../data), "
                             "hoặc nhiều ảnh là các trang của cùng một giấy")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Thư mục kết quả, mỗi giấy một thư mục con (mặc định: output_flow)")
    args = parser.parse_args()
    sources, output_dir = args.source, args.output_dir.expanduser().resolve()
    if len(sources) == 1 and sources[0].is_dir():
        return 1 if run_batch(sources[0], output_dir) else 0
    if any(p.is_dir() for p in sources):
        parser.error("Khi truyền nhiều đầu vào, tất cả phải là file.")
    try:
        run_flow(sources if len(sources) > 1 else sources[0], output_dir / sources[0].stem)
    except Exception:  # noqa: BLE001 - lỗi đã ghi log trong run_flow
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
