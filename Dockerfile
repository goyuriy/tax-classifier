FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (~400MB vs ~2.5GB for default/CUDA).
# sentence-transformers then uses this instead of pulling full PyTorch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install API deps only (no ipykernel/ipywidgets).
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY src/ src/
COPY models/ models/
COPY static/ static/
COPY api.py .
COPY main.py .

ENV PORT=8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
