# Educational demo container for a web-facing AI agent
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash tini curl iproute2 procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy server application files
COPY server/ /app/server/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD curl -fsS http://localhost:8000/health || exit 1

# Use tini as PID 1 to handle signals correctly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run FastAPI app with uvicorn on 0.0.0.0:8000
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
