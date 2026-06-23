FROM python:3.11-slim

# System deps:
#   libgl1 + libglib2.0-0 — OpenCV (EasyOCR)
#   libsm6 libxext6 libxrender1 — OpenCV headless libs
#   poppler-utils — pdf2image PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download trained YOLO weights from Hugging Face Hub at build time
RUN mkdir -p runs/detect/train/weights && \
    pip install huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ukers/artistmatch-model', filename='best.pt', local_dir='runs/detect/train/weights')"

EXPOSE 7860

CMD ["python", "app.py"]
