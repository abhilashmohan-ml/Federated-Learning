FROM python:3.12-slim

WORKDIR /app
COPY requirements/base.txt   requirements/base.txt
COPY requirements/client.txt requirements/client.txt

RUN pip install --no-cache-dir -r requirements/client.txt

COPY shared/ shared/
COPY client/ client/

ENV PYTHONPATH=/app
ARG FLET_PORT=8551
EXPOSE ${FLET_PORT}

CMD ["python", "-m", "client.main"]
