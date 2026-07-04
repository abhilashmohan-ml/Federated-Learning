FROM python:3.11-slim

WORKDIR /app
COPY requirements/base.txt   requirements/base.txt
COPY requirements/server.txt requirements/server.txt

RUN pip install --no-cache-dir -r requirements/server.txt

COPY shared/ shared/
COPY server/ server/

ENV PYTHONPATH=/app
EXPOSE 8000 8550

CMD ["sh", "-c",   "uvicorn server.main:app --host 0.0.0.0 --port 8000 &    python -m server.ui.app"]
