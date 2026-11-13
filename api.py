"""API HTTP: POST /ocr nhận file ảnh/PDF → trả JSON trích xuất.

Chạy: python run_api.py   (hoặc uvicorn api:app --host 0.0.0.0 --port 8002)
Gọi:  curl -F "file=@a.jpg" http://localhost:8002/ocr
"""

from functools import lru_cache
from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from extract_info import load_templates
from logger import logger
from main import SUPPORTED_SUFFIXES, run_flow
from qwen_ocr import QwenOCR

app = FastAPI(title="OCR GPKD")


@lru_cache(maxsize=1)
def _resources():
    """Dùng chung engine OCR và template cho mọi request."""
    return QwenOCR(), load_templates()


@app.get("/health")
def health():
    return {"status": "ok"}


# def (không async): run_flow chạy đồng bộ, FastAPI tự đưa vào threadpool.
@app.post("/ocr")
def ocr(file: UploadFile = File(...),
        include_text: bool = Query(False, description="Trả kèm text OCR từng trang")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Định dạng không hỗ trợ: {suffix or '(không có đuôi)'}; "
                                 f"chấp nhận {sorted(SUPPORTED_SUFFIXES)}")
    engine, templates = _resources()
    with tempfile.TemporaryDirectory(prefix="ocr_api_") as tmp:
        # Giữ tên file gốc: run_flow dùng tên file làm idRequest trong log.
        source = Path(tmp) / (Path(file.filename).name or f"upload{suffix}")
        with source.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"API nhận file {file.filename} ({source.stat().st_size} byte)")
        try:
            result = run_flow(source, engine, templates)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:  # noqa: BLE001 - đã ghi log trong run_flow
            raise HTTPException(500, f"Xử lý thất bại: {e}") from e
    body = {"filename": file.filename, "pages": len(result["pages"]), "extracted": result["extracted"]}
    if include_text:
        body["text"] = [{"page": p["page"], "text": p["text"]} for p in result["pages"]]
    return body
