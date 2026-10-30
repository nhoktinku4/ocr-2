"""HTTP API cho flow PDF/ảnh → OCR → trích xuất JSON."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from main import SUPPORTED_SUFFIXES, run_flow


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
FLOW_LOCK = Lock()
logger = logging.getLogger(__name__)
app = FastAPI(title="OCR GPKD API", version="1.0.0")


@app.get("/health")
def health():
    """Kiểm tra API đang chạy; không kiểm tra kết nối vLLM."""
    return {"status": "ok"}


@app.post("/ocr")
def ocr(
    file: Annotated[UploadFile, File(description="PDF hoặc ảnh, tối đa 50 MiB")],
    template: Annotated[
        Literal["auto", "doanh_nghiep", "ho_kinh_doanh"], Form()
    ] = "auto",
):
    """Trả trực tiếp JSON trích xuất; không lưu ảnh hoặc kết quả OCR."""
    filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, "Chỉ hỗ trợ PDF hoặc ảnh PNG/JPG/BMP/WebP/TIFF.")
    if not FLOW_LOCK.acquire(blocking=False):
        raise HTTPException(503, "Đang xử lý tài liệu khác, vui lòng thử lại.",
                            headers={"Retry-After": "5"})
    try:
        template_path = None if template == "auto" else ROOT / "json" / f"{template}.json"
        with TemporaryDirectory(prefix="ocr_upload_") as directory:
            source = Path(directory) / f"document{suffix}"
            size = 0
            with source.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "File vượt quá giới hạn 50 MiB.")
                    target.write(chunk)
            if not size:
                raise HTTPException(400, "File tải lên rỗng.")
            result = run_flow(source, Path(directory) / "output", template_path,
                              save_artifacts=False)
        return JSONResponse(result["extracted"], headers={
            "Content-Disposition": 'attachment; filename="extract.json"',
        })
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (RuntimeError, ImportError) as exc:
        logger.exception("OCR dependency failed")
        raise HTTPException(503, "Không thể chạy OCR; kiểm tra model và dịch vụ vLLM.") from exc
    except Exception as exc:
        logger.exception("OCR request failed")
        raise HTTPException(500, "Xử lý OCR thất bại; xem log server.") from exc
    finally:
        file.file.close()
        FLOW_LOCK.release()
 