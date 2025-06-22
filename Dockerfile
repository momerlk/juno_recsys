# Use official Python slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for LightFM, numpy, scipy etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements file first for caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . /app/

# Expose port (optional, for documentation; Cloud Run uses 8080 by default)
EXPOSE 8080

# Command to run the app
CMD ["gunicorn", "-w", "2", "--threads", "2", "-b", "0.0.0.0:8080", "main:app", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "300"]
