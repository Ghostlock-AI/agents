FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    git \
    curl \
    bash

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Just drop into shell, user does everything
CMD ["/bin/bash"]
