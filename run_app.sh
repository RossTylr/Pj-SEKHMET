#!/bin/bash
# Run SEKHMET Recovery Predictor App

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run ./setup_env.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Navigate to predictor directory and run Streamlit
cd src/predictor
echo "🚀 Starting SEKHMET Recovery Predictor..."
echo "📍 Running from: $(pwd)"
echo ""
streamlit run app.py

