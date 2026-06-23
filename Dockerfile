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

# Bundle trained YOLO weights.
# REQUIRES: runs/detect/train/weights/best.pt must exist locally before running docker build.
# This file is not committed to git — obtain it separately (see README.md).
COPY runs/detect/train/weights/best.pt runs/detect/train/weights/best.pt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
