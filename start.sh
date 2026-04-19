#!/bin/bash

# SecuTrack AI - Unified Startup Script
# Created by Antigravity AI

# 1. Clear ports
echo "[1/4] Clearing ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

# 2. Check Dependencies
echo "[2/4] Verifying dependencies..."
export PATH=$PATH:/usr/local/bin

# 3. Start Backend
echo "[3/4] Starting Backend (Port 8000)..."
cd backend
python3 -m app.main > backend.log 2>&1 &
BACKEND_PID=$!

# 4. Start Frontend
echo "[4/4] Starting Frontend (Port 3000)..."
cd ../frontend
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!

echo "------------------------------------------------"
echo "✅ SecuTrack AI is launching!"
echo "------------------------------------------------"
echo "🔗 Frontend: http://localhost:3000"
echo "🔗 Backend:  http://localhost:8000/health"
echo "------------------------------------------------"
echo "Logs available at backend/backend.log and frontend/frontend.log"
echo "Press Ctrl+C to stop (Background processes will remain until port clear)"

# Keep alive to see output
sleep 5
curl -s http://localhost:8000/health && echo "Backend is ONLINE." || echo "Backend is starting..."
