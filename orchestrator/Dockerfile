FROM python:3.12-slim

WORKDIR /app

COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator/server.py .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8800"]
