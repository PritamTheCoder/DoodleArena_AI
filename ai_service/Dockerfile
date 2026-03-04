# ai_service/Dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV (headless)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start both services: FastAPI (background) + Streamlit (foreground)
CMD bash -c "uvicorn main:app --host 0.0.0.0 --port 8000 --reload & streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"