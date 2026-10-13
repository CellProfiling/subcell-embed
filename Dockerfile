FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/subcell-embed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
