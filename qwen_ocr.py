#!/usr/bin/env python3
"""OCR từng vùng ảnh qua API OpenAI tương thích của vLLM."""

from __future__ import annotations

import base64
import io
import os
import math
import sys
import httpx

DEFAULT_MAX_PIXELS = 1_605_632
OCR_PROMPT = (
    "Perform OCR to extract all text from the document image.\n"
    "Preserve the original reading order, headings, hierarchy, paragraphs, Vietnamese text, and numerical data.\n"
    "Return null if no text in image.\n"
    "Return the result without any explanations."
)
TABLE_OCR_PROMPT = (
    "Transcribe this table image into a Markdown table, preserving exact rows and columns.\n"
    "Keep Vietnamese text, numbers, and symbols exact. Leave empty cells blank and do not duplicate merged cells.\n"
    "Output ONLY the Markdown table, no explanations."
)


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
            base_url=base_url or os.getenv("QWEN_OCR_BASE_URL", "http://127.0.0.1:8002/v1"),
            api_key=os.getenv("QWEN_OCR_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(
                trust_env=False
            )
        )

    def ocr_image(self, image, prompt: str | None = None, max_new_tokens: int = 2048,
                  *, max_time: float | None = None) -> str:
        """OCR ảnh PIL. max_time là timeout HTTP, lỗi timeout không trả text dở dang."""
        from openai import OpenAIError

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens phải lớn hơn 0.")
        if max_time is not None and (not math.isfinite(max_time) or max_time <= 0):
            raise ValueError("max_time phải là số hữu hạn lớn hơn 0.")
        prompt = OCR_PROMPT if prompt is None else prompt
        if not prompt.strip():
            raise ValueError("Prompt không được để trống.")
        image = image.convert("RGB")
        if image.width * image.height > self.max_pixels:
            scale = math.sqrt(self.max_pixels / (image.width * image.height))
            image = image.resize((max(1, int(image.width * scale)),
                                  max(1, int(image.height * scale))))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        options = {"timeout": max_time} if max_time is not None else {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=max_new_tokens,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                **options,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"Không thể OCR qua vLLM ({self.client.base_url}): {exc}") from exc
        if not response.choices or response.choices[0].message.content is None:
            raise RuntimeError("vLLM trả về phản hồi không có nội dung OCR.")
        if response.choices[0].finish_reason == "length":
            print("Cảnh báo: kết quả có thể bị cắt. Tăng max_new_tokens nếu cần.", file=sys.stderr)
        return response.choices[0].message.content.strip()
 