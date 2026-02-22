# Railway deployment — Python 3.11 + ffmpeg
FROM python:3.11-slim

# Install ffmpeg via apt (guaranteed available in debian-slim)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create temp directory for audio processing
RUN mkdir -p temp

CMD ["python", "bot.py"]
