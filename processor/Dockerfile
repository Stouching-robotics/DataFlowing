# ── Stage 1: Build React frontend ──
FROM node:22-alpine AS frontend
WORKDIR /src
COPY web/workflow-studio/package.json web/workflow-studio/package-lock.json* ./
RUN npm ci
COPY web/workflow-studio/ ./
RUN npm run build

# ── Stage 2: Python runtime ──
FROM python:3.12-slim
WORKDIR /srv/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web/templates ./web/templates
COPY web/static ./web/static
COPY --from=frontend /static/workflow-studio ./web/static/workflow-studio

EXPOSE 2586

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:2586/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2586"]
