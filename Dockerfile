# Optional production Dockerfile for Render (`runtime: docker`) with Node.js + Python ML support
FROM node:22-bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node.js production dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Install Python ML/Harvester dependencies in isolated venv
COPY requirements.txt ./
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"
ENV NODE_ENV=production
ENV PORT=10000

COPY . .

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
  && chown -R appuser:appgroup /app

USER appuser

EXPOSE 10000

CMD ["node", "src/server.js"]
