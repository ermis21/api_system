# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Install system dependencies required for PyTorch and transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget g++ python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Check if CUDA-capable GPU is present

RUN wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-debian12-12-6-local_12.6.2-560.35.03-1_amd64.deb
RUN dpkg -i cuda-repo-debian12-12-6-local_12.6.2-560.35.03-1_amd64.deb 
RUN cp /var/cuda-repo-debian12-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ 
RUN apt-get update && apt-get install -y software-properties-common 
RUN add-apt-repository -y contrib 
RUN apt-get update && apt-get install -y cuda-toolkit-12-6
RUN rm -f cuda-repo-debian12-12-6-local_12.6.2-560.35.03-1_amd64.deb; 
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV HUGGING_FACE_HUB_TOKEN="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"

# RUN nvcc --version

# Set the working directory in the container
WORKDIR /app

# fffffffffff
# fffffffff
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN pip install flash-attn

# Set environment variables for cache directories
ENV HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache

####
COPY app.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]