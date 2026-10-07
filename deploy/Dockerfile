# GWAYA benchmark runtime: Ollama (GPU-capable) + bubblewrap + Python 3.12 project deps.
# Model weights and the MBPP dataset are NOT baked in: they are restored from the SocrateAI data
# lake at run time, so one image serves any model/dataset revision. Built with Cloud Build
# (see cloudbuild.yaml); the build context is the extracted working-tree tarball from the lake.
FROM ollama/ollama:0.5.7

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends bubblewrap git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /usr/local/bin/uv

WORKDIR /app
# Dependency layer first (cache-friendly). CPU torch: the benchmark never uses torch for compute,
# it is only imported by gwaya/__init__.py, and the CUDA wheels would add several GB.
COPY pyproject.toml ./
RUN uv venv --python 3.12 /opt/venv \
 && VIRTUAL_ENV=/opt/venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && VIRTUAL_ENV=/opt/venv uv pip install -r pyproject.toml pandas pyarrow radon requests huggingface_hub

COPY . /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    OLLAMA_MODELS=/data/ollama/models \
    OLLAMA_HOST=127.0.0.1:11434
RUN chmod +x /app/deploy/container_entry.sh
ENTRYPOINT ["/app/deploy/container_entry.sh"]
