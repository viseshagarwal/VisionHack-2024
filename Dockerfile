FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    espeak \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Download YOLOv8 model
RUN python setup.py

# Expose port
EXPOSE 8501

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run the application
CMD ["streamlit", "run", "main.py"]