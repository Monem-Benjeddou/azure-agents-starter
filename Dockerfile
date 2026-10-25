FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["agents-starter", "serve", "--host", "0.0.0.0", "--port", "8000"]
