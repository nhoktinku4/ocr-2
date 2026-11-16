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


def _load_templates() -> dict[str, str]:
    """Đọc json/<loại>.json thành chuỗi JSON gọn để nhúng vào prompt."""
    templates = {}
    for kind in TYPE_DESCRIPTIONS:
        with open(os.path.join(JSON_DIR, f"{kind}.json"), encoding="utf-8") as f:
            templates[kind] = json.dumps(json.load(f), ensure_ascii=False)
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
        "- ma_so_doanh_nghiep / ma_so_ho_kinh_doanh / so_dang_ky: only the code digits.\n"
        "- dang_ky_lan_dau: date of first registration; dang_ky_thay_doi: lan_thu = change number, "
        "ngay = date of that change. Dates as written, preferably dd/mm/yyyy.\n"
        "- ten_cong_ty.tieng_viet / tieng_nuoc_ngoai / viet_tat: company name in Vietnamese, "
        "foreign language, abbreviation.\n"
        "- dia_chi: address only; dien_thoai, fax, email, website: each in its own field.\n"
        "- von_dieu_le / von_kinh_doanh: capital as written in digits (e.g. \"1.000.000.000 đồng\").\n"
        "- nganh_nghe_chinh: ten = business line name, ma_nganh = its code.\n"
        "- Persons (chu_so_huu, nguoi_dai_dien, chu_ho): ho_ten = full name only (no \"Ông\", \"Bà\"); "
        "gioi_tinh; ngay_sinh; dan_toc; quoc_tich; loai_giay_to = ID document type (CCCD, CMND, Hộ chiếu); "
        "so_giay_to = ID number only; ngay_cap, noi_cap; dia_chi_thuong_tru and dia_chi_lien_lac = addresses only; "
        "chuc_danh = position (e.g. \"Giám đốc\"); phan_von_gop, ty_le = capital contribution and percentage.\n"
        "Rules:\n"
        "- Output a single JSON object following the identified template.\n"
        "- Keep exactly the template keys and structure; include every key, do not add or rename keys.\n"
        "- Use null for fields not present in the images; never guess or copy a value into another field.\n"
        "- Lists: one object per member / legal representative (members are often in a table).\n"
        "- The section \"Thông tin về chủ sở hữu\" (owner of a single-member LLC / private enterprise) "
        "goes into chu_so_huu (one object per owner), same as \"Danh sách thành viên góp vốn\" / "
        "\"Danh sách cổ đông sáng lập\".\n"
        "- nguoi_dai_dien: ONLY the person(s) in the section \"Người đại diện theo pháp luật\" "
        "(one object per representative); do not copy owners or members into it unless they are also "
        "listed in that section.\n"
        "- IGNORE the signature block at the end of the certificate (\"TRƯỞNG PHÒNG\", \"PHÓ TRƯỞNG PHÒNG\", "
        "\"KT. TRƯỞNG PHÒNG\", \"TM. ỦY BAN NHÂN DÂN\", stamp/seal, signer's name): the signer is an official, "
        "never put that name in chu_so_huu, nguoi_dai_dien or chu_ho.\n"
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
