# backend/Dockerfile
FROM python:3.13-slim

WORKDIR /app

# 1. Install system dependencies (git is needed for CLIP)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. OPTIMIZATION: Install heavy libraries ALONE first
# This prevents RAM spikes during dependency resolution
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Copy requirements
COPY requirements.txt /app/backend/requirements.txt

# 4. Install the rest
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 5. Copy backend code
COPY . /app/backend

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]