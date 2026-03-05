FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY fixtures/ fixtures/

RUN adduser --disabled-password --gecos '' appuser \
    && mkdir -p /app/logs /app/data && chown appuser:appuser /app/logs /app/data
USER appuser

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
