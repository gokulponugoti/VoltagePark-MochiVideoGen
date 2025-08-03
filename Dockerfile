FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Python and essential packages
RUN apt update && apt install -y \
    python3 python3-pip \
    git curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variable (for runtime mount)
ENV MOCHI_WEIGHTS_DIR=/weights

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy mochi_src and install it
COPY mochi_src ./mochi_src
WORKDIR /app/mochi_src
RUN pip3 install ./src
WORKDIR /app

# Copy core app code
COPY main.py model_runner.py ./
COPY app ./app

# Expose the API port
EXPOSE 8080

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
