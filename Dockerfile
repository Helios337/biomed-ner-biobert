FROM python:3.10-slim

WORKDIR /app

# Install system deps for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml requirements.txt ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e "."

# Default: run inference (mount a saved model at /app/models)
CMD ["python", "-c", "from src.inference import BioNERPipeline; p = BioNERPipeline(); print(p.predict('Patient with colorectal cancer and diabetes'))"]