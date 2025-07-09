
echo "🏥 Starting Insurance Agent System..."

if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please create it with your API keys."
    echo "📝 Copy .env.example to .env and fill in your API keys:"
    echo "   cp .env.example .env"
    exit 1
fi

if [ ! -f "insurance_agent.py" ]; then
    echo "❌ insurance_agent.py not found!"
    exit 1
fi

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
pkill -f "python token_server.py" 2>/dev/null || true
pkill -f "python insurance_agent.py" 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start token server
echo "🚀 Starting token server on port 8000..."
python token_server.py &
TOKEN_PID=$!

# Wait for token server to start
sleep 3

# Test token server
echo "🧪 Testing token server..."
python -c "
import requests
try:
    response = requests.get('http://localhost:8000/api/insurance/health', timeout=5)
    if response.status_code == 200:
        print('✅ Token server is running')
    else:
        print('❌ Token server health check failed')
        exit(1)
except Exception as e:
    print(f'❌ Cannot connect to token server: {e}')
    exit(1)
" || {
    echo "❌ Token server failed to start"
    kill $TOKEN_PID 2>/dev/null || true
    exit 1
}

# Start insurance agent in console mode
echo "🤖 Starting insurance agent in console mode..."
echo ""
echo "🎯 Insurance Agent Console Ready!"
echo ""
echo "📋 System Status:"
echo "   🔗 Token Server: http://localhost:8000 (Running)"
echo "   🤖 Insurance Agent: Console Mode (Starting...)"
echo ""
echo "🗣️  You can now speak to the insurance agent!"
echo "💡 Say things like:"
echo "   - 'Hello, I need health insurance'"
echo "   - 'I want coverage for my family'"
echo "   - 'Tell me about critical illness plans'"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping Insurance System..."
    kill $TOKEN_PID 2>/dev/null || true
    pkill -f "python token_server.py" 2>/dev/null || true
    pkill -f "python insurance_agent.py" 2>/dev/null || true
    echo "✅ Insurance System stopped."
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Start insurance agent and wait
python insurance_agent.py console

# If we get here, the agent stopped
cleanup