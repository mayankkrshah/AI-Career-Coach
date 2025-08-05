#!/bin/bash

# Career Coach App Stop Script
echo "🛑 Stopping Career Coach App..."

# Kill processes on ports
echo "🧹 Stopping backend on port 9000..."
lsof -ti:9000 | xargs kill -9 2>/dev/null || true

echo "🧹 Stopping frontend on port 4002..."
lsof -ti:4002 | xargs kill -9 2>/dev/null || true

# Also kill any node processes running dev
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true

# Kill any python processes running the app
pkill -f "simple_app.py" 2>/dev/null || true
pkill -f "career_coach_app.py" 2>/dev/null || true

echo "✅ All processes stopped!"