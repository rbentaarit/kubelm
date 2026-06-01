# kubelm-agent: drives kubelm through K8sGPT's MCP tools.
# Build from the repo root:  docker build -f deploy/agent/Dockerfile -t kubelm-agent:dev .
FROM python:3.11-slim

WORKDIR /app

# The agent path needs only the core runtime deps (no training/ML stack).
RUN pip install --no-cache-dir "requests>=2.31" "jsonschema>=4.21" "pyyaml>=6.0"

# Reuse the eval run-loop + MCP client + backend; the agent is a thin wrapper.
COPY eval/ /app/eval/
COPY deploy/agent/server.py /app/server.py

ENV PORT=8080
EXPOSE 8080

CMD ["python", "/app/server.py"]
