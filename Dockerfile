FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY egoalign ./egoalign
RUN pip install --no-cache-dir ".[align]"
ENV EGOALIGN_OUTPUT=/data EGOALIGN_CACHE=/cache HF_HOME=/cache/hf
VOLUME ["/data", "/cache"]
EXPOSE 8000
CMD ["egoalign", "serve", "--host", "0.0.0.0", "--port", "8000"]
