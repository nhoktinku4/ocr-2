FROM python:3.12-slim

# poppler-utils: pdf2image đọc PDF
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch từ PyPI trên Linux đã kèm CUDA runtime; chỉ cần driver NVIDIA + nvidia-container-toolkit
# trên máy chủ. Không có GPU thì table_detection.py tự chạy CPU.
RUN pip install --no-cache-dir \
    torch==2.15.0 \
    transformers==5.22.1 \
    opencv-python-headless==5.0.0.93 \
    numpy==2.3.5 \
    pillow==13.0.0 \
    pdf2image==1.17.0 \
    pymupdf==2.1 \
    httpx==0.28.1 \
    openai==3.49.0

# COPY json/ json/
# COPY *.py ./

ENV QWEN_OCR_BASE_URL=http://host.docker.internal:8002/v1 \
    PYTHONUNBUFFERED=1
# Chạy: docker compose run --rm ocr <file hoặc thư mục trong source/>; kết quả trích xuất ghi vào log (/app/logs).
# Batch không mở cổng nên không có /health; kiểm tra server OCR (vLLM) còn trả lời GET /models.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(os.environ['QWEN_OCR_BASE_URL'].rstrip('/') + '/models', timeout=4)" || exit 1
ENTRYPOINT ["python", "main.py"]
