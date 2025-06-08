FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir scikit-learn matplotlib scipy numpy && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
