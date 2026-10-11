# Gateway image for the GWAYA/GwenLaya serverless endpoint (CPU only; the TPU VM is created on demand).
# Small on purpose: no model, no CUDA, so the Cloud Run cold start of the gateway itself is ~1-2 s.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir "fastapi>=0.110,<1" "uvicorn[standard]>=0.29,<1" "httpx>=0.27,<1"
WORKDIR /app
COPY gateway/main.py /app/main.py
COPY vm /app/vm
ENV VM_SCRIPTS_DIR=/app/vm
# single worker: the TPU lifecycle lock is per process; Cloud Run scales instances for HTTP load
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --no-access-log"]
