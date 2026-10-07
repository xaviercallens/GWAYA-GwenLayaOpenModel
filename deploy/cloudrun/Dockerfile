# GwenLaya generation endpoint for Cloud Run GPU (L4): llama.cpp server (CUDA) + thin FastAPI front.
# Engine choice: llama.cpp, because the preregistered tiers are q4_K_M GGUF files (plan.json S6/S7)
# and S1 measures llama.cpp --parallel 8. vLLM is not used (no GGUF-first path, bigger image).
# The base tag is a moving tag: pin a digest after the first successful build (recorded as TBD).
ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
FROM ${LLAMA_CPP_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "fastapi>=0.110,<1" "uvicorn>=0.29,<1" "httpx>=0.27,<1"

WORKDIR /app
COPY front.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Weights are NOT in the image by default: a Cloud Storage volume is mounted at /models (deploy.sh).
# To bake instead (avoids cross-region reads, see docs/CLOUDRUN_ENDPOINT.md), COPY a GGUF to /models.
ENV MODEL_DIR=/models LLAMA_PORT=8081 LLAMA_PARALLEL=8 LLAMA_CTX=8192 LLAMA_NGL=99
ENTRYPOINT ["/app/entrypoint.sh"]
