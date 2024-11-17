# Use the official Ollama base image
FROM ollama/ollama:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python and necessary dependencies
# Assuming the Ollama image is based on a Debian-like distro
RUN apt-get update && \
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Set work directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

# Copy the nanograd project into the container
COPY nanograd/ /app/nanograd

# Expose the port that Gradio will run on (default: 7860)
EXPOSE 7860

# Set the entrypoint to run engine.py
CMD ["python3", "nanograd/engine.py"]
