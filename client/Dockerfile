FROM python:3.11-slim

WORKDIR /app
COPY requirements/base.txt   requirements/base.txt
COPY requirements/client.txt requirements/client.txt

RUN pip install --no-cache-dir -r requirements/client.txt

COPY shared/ shared/
COPY client/ client/

ENV PYTHONPATH=/app
EXPOSE 8551

CMD ["python", "-m", "client.main"]
