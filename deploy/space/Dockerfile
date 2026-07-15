FROM python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HARBOR_HF_WEB_DIR=/app/apps/results-web/dist

WORKDIR /app

RUN pip install --no-cache-dir uv==0.8.22

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv export --locked --no-dev --no-group presentation --no-emit-project \
      --output-file /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && uv pip install --system --no-cache --no-deps .

COPY apps/results-web/dist ./apps/results-web/dist

EXPOSE 7860

CMD ["uvicorn", "harbor_hf.presentation.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
