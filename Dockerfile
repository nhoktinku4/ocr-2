FROM python:3.12-slim

# poppler-utils: pdf2image đọc PDF
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model heron không nằm trong git: tải từ Hugging Face lúc build
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('docling-project/docling-layout-heron', local_dir='docling-layout-heron', \
allow_patterns=['config.json', 'model.safetensors', 'preprocessor_config.json'])"

COPY json/ json/
COPY *.py ./

ENV QWEN_OCR_BASE_URL=http://host.docker.internal:8002/v1 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
# 1 worker: api.py chỉ xử lý một tài liệu một lúc (FLOW_LOCK)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
