FROM python:3.12-slim

WORKDIR /app
COPY requirements/base.txt   requirements/base.txt
COPY requirements/server.txt requirements/server.txt

RUN pip install --no-cache-dir -r requirements/server.txt

COPY shared/   shared/
COPY server/   server/
COPY scripts/  scripts/

ENV PYTHONPATH=/app
EXPOSE 8000 8550

# init_db.py is idempotent — safe to run on every startup.
# Flet dashboard runs in foreground; uvicorn runs in background.
CMD ["sh", "-c", "python scripts/init_db.py && python server/main.py & python -m server.ui.app"]
