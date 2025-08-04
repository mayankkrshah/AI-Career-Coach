#!/bin/bash

# Career Coach App Startup Script
echo "🚀 Starting Career Coach App..."

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
lsof -ti:9000 | xargs kill -9 2>/dev/null || true
lsof -ti:4002 | xargs kill -9 2>/dev/null || true

# Start backend
echo "📡 Starting Backend API on port 9000..."
cd api
nohup python3 simple_app.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:9000/api/health > /dev/null; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start. Check api/backend.log"
    exit 1
fi

# Start frontend
echo "🎨 Starting Frontend on port 4002..."
cd ../frontend
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 10

echo "✅ App is running!"
echo "🌐 Open http://localhost:4002 in your browser"
echo ""
echo "📝 Logs:"
echo "  Backend: api/backend.log"
echo "  Frontend: frontend/frontend.log"
echo ""
echo "🛑 To stop: kill $BACKEND_PID $FRONTEND_PID"