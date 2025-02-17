#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker and Docker Compose if not installed
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create app directory
mkdir -p /home/ubuntu/business-plan-app
cd /home/ubuntu/business-plan-app

# Set up environment variables
echo "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" > .env
echo "HUGGINGFACE_MODEL_ID=${HUGGINGFACE_MODEL_ID}" >> .env

# Pull the latest code (you'll need to set up your git credentials)
git pull origin main

# Build and start the containers
docker-compose -f docker-compose.yml up -d --build

# Print the status
docker-compose ps