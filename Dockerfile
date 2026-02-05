FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY models/ models/
COPY static/ static/
COPY api.py .
COPY main.py .

# Set default port
ENV PORT=8000

# Start the application
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
