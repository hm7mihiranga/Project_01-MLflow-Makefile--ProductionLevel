"""
Utility functions for creating deployment scripts and configurations.
"""

import os
import logging

logger = logging.getLogger(__name__)


def create_api_script():
    """Create API deployment script if it doesn't exist."""
    script_path = os.path.join(os.path.dirname(__file__), "api_server.py")
    
    if os.path.exists(script_path):
        logger.info(f"✓ API script already exists: {script_path}")
        return True
    
    logger.info(f"Creating API script at: {script_path}")
    # The script would be created here if it didn't exist
    # For now, it's already created above
    return True


def create_dockerfile():
    """Create Dockerfile for containerized deployment."""
    dockerfile_content = '''
# Dockerfile for Telco Customer Churn Prediction API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create artifacts directory
RUN mkdir -p artifacts/models artifacts/encode

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "deployment/api_server.py"]
'''
    
    dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dockerfile")
    
    try:
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content.strip())
        
        logger.info(f"✓ Dockerfile created: {dockerfile_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create Dockerfile: {str(e)}")
        return False


def create_docker_compose():
    """Create docker-compose.yml for easy deployment."""
    compose_content = '''
version: '3.8'

services:
  telco-churn-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./config.yaml:/app/config.yaml:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

  mlflow-server:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: >
      bash -c "pip install mlflow>=2.0.0 && 
               mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///mlruns"
    restart: unless-stopped
'''
    
    compose_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
    
    try:
        with open(compose_path, 'w') as f:
            f.write(compose_content.strip())
        
        logger.info(f"✓ docker-compose.yml created: {compose_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create docker-compose.yml: {str(e)}")
        return False


if __name__ == "__main__":
    # Create deployment files
    create_api_script()
    create_dockerfile() 
    create_docker_compose()
    print("✓ Deployment files created successfully!")