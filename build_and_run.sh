#!/bin/bash
# Script to build and run the RAG application using Docker Compose

# Display header
echo "==================================================="
echo "  Building and Running RAG Documentation Assistant  "
echo "==================================================="

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose down

# Build the image
echo "Building Docker image..."
docker-compose build

# Run the container
echo "Starting container..."
docker-compose up -d

# Check status
echo "Container status:"
docker-compose ps

# Display logs
echo "=================================================="
echo "Container is now running at http://localhost:8000"
echo "To view logs, run: docker-compose logs -f"
echo "To stop the container, run: docker-compose down"
echo "=================================================="
