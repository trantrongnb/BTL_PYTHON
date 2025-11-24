# Base image nhẹ với Python 3.10
FROM python:3.10-slim

# Cài thư viện hệ thống cần cho video và OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    librsvg2-2 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài thư viện Python
RUN pip install --no-cache-dir \
    opencv-python \
    fastapi \
    uvicorn[standard] \
    jinja2 \
    python-multipart \
    torch \
    torchvision \
    ultralytics \
    numpy \
    pillow

# Thiết lập thư mục làm việc
WORKDIR /app
COPY . /app

# Mặc định mở Python
CMD ["python"]
