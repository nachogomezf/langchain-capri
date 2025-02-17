#!/bin/bash

# Set up SSL with Let's Encrypt
setup_ssl() {
    # Install certbot
    sudo apt-get install -y certbot python3-certbot-nginx

    # Get SSL certificate
    sudo certbot --nginx -d $DOMAIN_NAME --non-interactive --agree-tos -m $EMAIL --redirect

    # Reload nginx
    sudo systemctl reload nginx
}

# Main setup
echo "Setting up Business Plan Generator on Lightsail..."

# Install required packages
sudo apt-get update
sudo apt-get install -y git nginx python3-pip docker.io docker-compose

# Clone the repository
git clone https://github.com/your-username/langchain-business-plan.git /home/ubuntu/business-plan-app
cd /home/ubuntu/business-plan-app

# Copy environment variables
cp .env.prod .env

# Start the application
docker-compose up -d

# Set up SSL if domain is configured
if [ ! -z "$DOMAIN_NAME" ] && [ ! -z "$EMAIL" ]; then
    setup_ssl
fi

echo "Setup complete! Your application should be running at http://$DOMAIN_NAME"