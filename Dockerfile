# Multi-stage build for Causal Pricing Engine API
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# Train a fallback model artifact inside the image (synthetic data)
RUN python scripts/train.py

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src ./src
COPY --from=builder /build/configs ./configs
COPY --from=builder /build/scripts ./scripts
COPY --from=builder /build/artifacts ./artifacts
COPY --from=builder /build/pyproject.toml ./pyproject.toml
COPY --from=builder /build/README.md ./README.md

RUN chmod +x /app/scripts/docker_entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8100

ENV PYTHONUNBUFFERED=1
ENV CPE_API__PORT=8100

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
