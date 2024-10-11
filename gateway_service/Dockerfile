FROM python:3.9-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
COPY . .
RUN apt update && \
    apt install -y curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip --no-cache install -e .
