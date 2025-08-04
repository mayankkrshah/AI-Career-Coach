#!/bin/bash

# AI Career Coach Development Script
# This script starts the development environment

set -e

echo "🛠️  Starting AI Career Coach in development mode..."

# Check if Python virtual environment exists
if [ ! -d "./api/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    cd api
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
else
    echo "📦 Activating existing virtual environment..."
fi

# Check if .env exists
if [ ! -f "./api/.env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp ./api/.env.example ./api/.env
    echo "📝 Please edit ./api/.env with your API keys."
fi

# Create data directory
mkdir -p data

# Check if job dataset exists
if [ ! -f "./data/job_dataset.csv" ]; then
    echo "⚠️  Job dataset not found."
    echo "   Please add job_dataset.csv to the ./data/ directory."
    echo "   You can download a sample dataset or use your own job market data."
fi

echo ""
echo "🚀 Starting development servers..."
echo ""

# Start backend in background
echo "🔧 Starting backend API..."
cd api
source venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 9000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 5

# Start frontend in background
echo "🎨 Starting frontend..."
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Development servers started!"
echo ""
echo "📋 Service URLs:"
echo "   Frontend: http://localhost:4000"
echo "   API: http://localhost:9000"
echo "   API Docs: http://localhost:9000/docs"
echo ""
echo "🛑 To stop development servers:"
echo "   Press Ctrl+C or run: kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping development servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup INT

# Wait for user to stop
echo "Press Ctrl+C to stop the development servers..."
wait