#!/bin/bash

# AI Career Coach Deployment Script
# This script deploys the AI Career Coach application using Docker Compose

set -e

echo "🚀 Starting AI Career Coach deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data logs nginx/ssl

# Check if .env file exists
if [ ! -f "./api/.env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp ./api/.env.example ./api/.env
    echo "📝 Please edit ./api/.env with your API keys before running the application."
fi

# Check if job dataset exists
if [ ! -f "./data/job_dataset.csv" ]; then
    echo "⚠️  Job dataset not found. Please add job_dataset.csv to the ./data/ directory."
    echo "   You can download a sample dataset or use your own job market data."
fi

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose ps

# Test API endpoint
echo "🧪 Testing API health..."
if curl -f http://localhost:9000/api/health > /dev/null 2>&1; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed. Check logs with: docker-compose logs api"
fi

# Test frontend
echo "🧪 Testing frontend..."
if curl -f http://localhost:4000 > /dev/null 2>&1; then
    echo "✅ Frontend is accessible!"
else
    echo "❌ Frontend is not accessible. Check logs with: docker-compose logs frontend"
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Service URLs:"
echo "   Frontend: http://localhost:4000"
echo "   API: http://localhost:9000"
echo "   API Docs: http://localhost:9000/docs"
echo "   Qdrant: http://localhost:6333/dashboard"
echo ""
echo "📊 Useful commands:"
echo "   View logs: docker-compose logs -f [service_name]"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo "   Update services: docker-compose pull && docker-compose up -d"
echo ""
echo "📝 Don't forget to:"
echo "   1. Add your API keys to ./api/.env"
echo "   2. Upload job dataset to ./data/job_dataset.csv"
echo "   3. Load data via: curl -X POST 'http://localhost:9000/api/load-data?api_key=YOUR_OPENAI_KEY'"