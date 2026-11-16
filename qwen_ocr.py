#!/usr/bin/env python3
"""OCR GPKD qua API OpenAI tương thích của vLLM (Qwen-VL): ảnh các trang → JSON theo mẫu."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time

import httpx

from logger import logger

DEFAULT_MAX_PIXELS = 1_605_632
JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")
TYPE_DESCRIPTIONS = {
    "doanh_nghiep": 'Giấy chứng nhận đăng ký doanh nghiệp (công ty TNHH, cổ phần, doanh nghiệp tư nhân...; '
                    'có "Mã số doanh nghiệp", "Tên công ty", "Trụ sở chính", "Vốn điều lệ", '
                    '"Người đại diện theo pháp luật")',
    "ho_kinh_doanh": 'Giấy chứng nhận đăng ký hộ kinh doanh (có "Tên hộ kinh doanh", "Địa điểm kinh doanh", '
                     '"Ngành, nghề kinh doanh", "Vốn kinh doanh", "Chủ hộ kinh doanh" / "Người đại diện hộ kinh doanh")',
}


# Model chép lại cả bảng ngành nghề vào khóa này (giữ nguyên chữ "(chính)" ở mã ngành);
# main.py chọn ngành chính rồi điền nganh_nghe_chinh. Chọn trong code chắc hơn để model 4B tự chọn.
BUSINESS_LINES_KEY = "danh_sach_nganh_nghe"
# Model ghi tên người ký (cán bộ cấp giấy) vào khóa này thay vì nhét nhầm vào mục người;
# main.py dùng nó để lọc rồi bỏ khỏi kết quả.
SIGNER_KEY = "nguoi_ky"


def _load_templates() -> dict[str, str]:
    """Đọc json/<loại>.json thành chuỗi JSON gọn để nhúng vào prompt."""
    templates = {}
    for kind in TYPE_DESCRIPTIONS:
        with open(os.path.join(JSON_DIR, f"{kind}.json"), encoding="utf-8") as f:
            template = json.load(f)
        if "nganh_nghe_chinh" in template:
            template = {(BUSINESS_LINES_KEY if k == "nganh_nghe_chinh" else k):
                        ([{"ten": None, "ma_nganh": None}] if k == "nganh_nghe_chinh" else v)
                        for k, v in template.items()}
        template[SIGNER_KEY] = None
        templates[kind] = json.dumps(template, ensure_ascii=False)
    return templates


def build_ocr_prompt() -> str:
    """Prompt: nhận diện loại giấy, rồi trả về JSON theo mẫu tương ứng."""
    sections = "\n".join(
        f"- {kind}: {TYPE_DESCRIPTIONS[kind]}\n  Template: {tpl}"
        for kind, tpl in _load_templates().items()
    )
    return (
        "Perform OCR on this Vietnamese business registration certificate. "
        "You may receive one or several images: they are pages of the SAME certificate, in order. "
        "Read all pages together and return ONE combined JSON.\n"
        "Step 1: Identify the certificate type:\n"
        f"{sections}\n"
        "Step 2: Fill the JSON template of the identified type with values read from the images.\n"
        "Field meanings (put each value ONLY in its own field, never mix fields):\n"
        "- loai_giay_to: keep exactly the value given in the template.\n"
        "- co_quan_cap: issuing authority (\"Sở Tài chính / Sở Kế hoạch và Đầu tư <tỉnh>\", "
        "\"Phòng Đăng ký kinh doanh\", \"UBND quận/huyện ...\", \"Phòng Tài chính - Kế hoạch\").\n"
        "  co_quan_cap (doanh_nghiep): the header at the top-left is split into 3 fields:\n"
        "    so = \"Sở\" (department), NOT \"số\" (number): only the department name WITHOUT the province, "
        "e.g. \"Sở Kế hoạch và Đầu tư\", \"Sở Tài chính\";\n"
        "    phong = the office, e.g. \"Phòng Đăng ký kinh doanh\";\n"
        "    tinh_thanh = ONLY the province/city name after \"TỈNH\" / \"THÀNH PHỐ\", e.g. \"Bình Phước\".\n"
        "    Example: header \"SỞ KẾ HOẠCH VÀ ĐẦU TƯ TỈNH BÌNH PHƯỚC / PHÒNG ĐĂNG KÝ KINH DOANH\" → "
        "{\"so\": \"Sở Kế hoạch và Đầu tư\", \"phong\": \"Phòng Đăng ký kinh doanh\", \"tinh_thanh\": \"Bình Phước\"}.\n"
        "  co_quan_cap (ho_kinh_doanh): ubnd = e.g. \"UBND huyện X\", phong = e.g. \"Phòng Tài chính - Kế hoạch\".\n"
        "- ma_so_doanh_nghiep / ma_so_ho_kinh_doanh / so_dang_ky: only the code digits.\n"
        "- dang_ky_lan_dau: date of first registration; dang_ky_thay_doi: lan_thu = change number, "
        "ngay = date of that change. Dates as written, preferably dd/mm/yyyy.\n"
        "- ten_cong_ty.tieng_viet / tieng_nuoc_ngoai / viet_tat: company name in Vietnamese, "
        "foreign language, abbreviation.\n"
        "- dia_chi: address only; dien_thoai, fax, email, website: each in its own field.\n"
        "- von_dieu_le / von_kinh_doanh: capital as written in digits (e.g. \"1.000.000.000 đồng\").\n"
        f"- {BUSINESS_LINES_KEY}: copy EVERY row of the business lines table (\"Ngành, nghề kinh doanh\"), "
        "in order, one object per row: ten = the name in column \"Tên ngành\", ma_nganh = the text in column "
        "\"Mã ngành\" EXACTLY as written, KEEPING any word \"chính\" (e.g. \"4632\", \"4711 (Chính)\", "
        "\"4632(chính)\"). Do not choose or skip rows.\n"
        "- Persons (chu_so_huu, nguoi_dai_dien, chu_ho): ho_ten = full name only (no \"Ông\", \"Bà\"); "
        "gioi_tinh; ngay_sinh; dan_toc; quoc_tich; loai_giay_to = ID document type (CCCD, CMND, Hộ chiếu); "
        "so_giay_to_(CMND/CCCD) = the number of the person's ID / legal document: số CCCD (Căn cước công dân, "
        "12 digits), số CMND (Chứng minh nhân dân, 9 or 12 digits), số định danh cá nhân, or the number of any other "
        "legal document shown (Hộ chiếu, Giấy tờ pháp lý của cá nhân...); digits/characters of the number only, "
        "without the label; ngay_cap, noi_cap; dia_chi_thuong_tru and dia_chi_lien_lac = addresses only; "
        "chuc_danh = position (e.g. \"Giám đốc\"); phan_von_gop, ty_le = capital contribution and percentage.\n"
        "Rules:\n"
        "- Output a single JSON object following the identified template.\n"
        "- Keep exactly the template keys and structure; include every key, do not add or rename keys.\n"
        "- Use null for fields not present in the images; never guess or copy a value into another field.\n"
        "- Take each value ONLY from its own section/label on the certificate. If a section is missing or "
        "empty, use null: NEVER fill it with data from another section (e.g. do not use dia_chi_thuong_tru for "
        "dia_chi_lien_lac, the business address for the owner's address, or the owner for nguoi_dai_dien).\n"
        "- Lists: one object per member / legal representative (members are often in a table).\n"
        "- The section \"Thông tin về chủ sở hữu\" (owner of a single-member LLC / private enterprise) "
        "goes into chu_so_huu (one object per owner), same as \"Danh sách thành viên góp vốn\" / "
        "\"Danh sách cổ đông sáng lập\".\n"
        "- nguoi_dai_dien: ONLY the person(s) in the section \"Người đại diện theo pháp luật\" "
        "(one object per representative); do not copy owners or members into it unless they are also "
        "listed in that section.\n"
        f"- {SIGNER_KEY}: the name in the signature block at the end of the certificate (under \"TRƯỞNG PHÒNG\", "
        "\"PHÓ TRƯỞNG PHÒNG\", \"KT. TRƯỞNG PHÒNG\", \"TM. ỦY BAN NHÂN DÂN\", next to the stamp/seal). "
        "The signer is the issuing official: put that name ONLY in "
        f"{SIGNER_KEY}, NEVER in chu_so_huu, nguoi_dai_dien or chu_ho.\n"
        "- nguoi_dai_dien comes from the section \"Người đại diện theo pháp luật\" in the body, which lists "
        "chức danh, ngày sinh, giấy tờ, địa chỉ. A name that stands alone with no such details is NOT a "
        "representative.\n"
        "- Preserve Vietnamese diacritics and numbers exactly as written.\n"
        "- Return only the JSON, without markdown or explanations."
    )


# Prompt nhận diện loại giấy + trả JSON theo mẫu.
OCR_PROMPT = build_ocr_prompt()


class QwenOCR:
    """Gọi model Qwen được phục vụ bởi vLLM qua OpenAI Chat Completions."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 max_pixels: int = DEFAULT_MAX_PIXELS, timeout: float = 120.0) -> None:
        from openai import OpenAI

        if max_pixels < 65_536:
            raise ValueError("max_pixels phải từ 65536 trở lên.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout phải là số hữu hạn lớn hơn 0.")
        self.max_pixels = max_pixels
        self.model = model or os.getenv("QWEN_OCR_MODEL", "qwen-ocr")
        self.client = OpenAI(
            base_url=base_url or os.getenv("QWEN_OCR_BASE_URL", "http://127.0.0.1:8007/v1"),
            api_key=os.getenv("QWEN_OCR_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )

    def _encode(self, image, max_pixels: int) -> str:
        image = image.convert("RGB")
        if image.width * image.height > max_pixels:
            scale = math.sqrt(max_pixels / (image.width * image.height))
            image = image.resize((max(1, int(image.width * scale)),
                                  max(1, int(image.height * scale))))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def ocr_images(self, images, prompt: str | None = None, max_new_tokens: int = 2048,
                   *, max_time: float | None = None, max_total_pixels: int | None = None) -> str:
        """Gửi nhiều ảnh PIL trong MỘT request (các trang của một giấy) → một câu trả lời.

        max_total_pixels: tổng số pixel cho mọi ảnh (chia đều), để không vượt context của model.
        vLLM cần --limit-mm-per-prompt '{"image": N}' với N >= số ảnh.
        """
        from openai import OpenAIError

        images = list(images)
        if not images:
            raise ValueError("Cần ít nhất một ảnh.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens phải lớn hơn 0.")
        if max_time is not None and (not math.isfinite(max_time) or max_time <= 0):
            raise ValueError("max_time phải là số hữu hạn lớn hơn 0.")
        prompt = OCR_PROMPT if prompt is None else prompt
        if not prompt.strip():
            raise ValueError("Prompt không được để trống.")
        per_image = self.max_pixels
        if max_total_pixels:
            per_image = max(65_536, min(per_image, max_total_pixels // len(images)))
        started = time.perf_counter()
        encoded = [self._encode(im, per_image) for im in images]
        logger.info(f"Đã mã hóa {len(images)} ảnh (≤{per_image:,} px/ảnh, {sum(map(len, encoded)) / 1e6:.1f} MB base64) "
                    f"trong {time.perf_counter() - started:.1f}s; gửi tới {self.client.base_url}")
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{e}"}} for e in encoded]
        options = {"timeout": max_time} if max_time is not None else {}
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}],
                max_tokens=max_new_tokens,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                **options,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"Không thể OCR qua vLLM ({self.client.base_url}): {exc}") from exc
        logger.info(f"vLLM trả lời sau {time.perf_counter() - started:.1f}s, usage={getattr(response, 'usage', None)}")
        if not response.choices or response.choices[0].message.content is None:
            raise RuntimeError("vLLM trả về phản hồi không có nội dung OCR.")
        if response.choices[0].finish_reason == "length":
            logger.warning("Kết quả OCR có thể bị cắt (finish_reason=length). Tăng max_new_tokens nếu cần.")
        return response.choices[0].message.content.strip()
