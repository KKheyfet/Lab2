# Use the NVIDIA L4T Base image
FROM nvcr.io/nvidia/l4t-base:r36.2.0

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install MediaPipe
RUN pip3 install mediapipe

# Set the working directory
WORKDIR /app

# Copy your pose estimation script into the container
COPY pose_estimation.py /app/pose_estimation.py

# Command to run the script
CMD ["python3", "pose_estimation.py"]
