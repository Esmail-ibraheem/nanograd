# Use an official Python image from the Docker Hub
FROM python:3.8-slim

# Set a working directory for the app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install torch manually (CPU or CUDA version)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Gradio, Pillow, and transformers
RUN pip install gradio pillow transformers

# Copy the app files into the container
COPY . .

# Ensure the required folders and model files exist in the correct location
RUN mkdir -p /app/nanograd/models/stable_diffusion/sd_data

# Add any necessary additional commands to download model files
# Example: If you're fetching a model from a URL
# RUN wget -O /app/nanograd/models/stable_diffusion/sd_data/v1-5-pruned-emaonly.ckpt <MODEL_URL>

# Expose the port Gradio will run on
EXPOSE 7860

# Set environment variables
ENV DEVICE=cpu \
    ALLOW_CUDA=False \
    ALLOW_MPS=True

# Command to run the engine (this would be your main script)
CMD ["python", "main.py"]
