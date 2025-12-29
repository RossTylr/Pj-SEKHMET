#!/bin/bash
# Environment Setup Script for Pj-SEKHMET
# Creates virtual environment and installs dependencies

set -e

echo "🔧 Setting up Python environment..."

# Create virtual environment
echo "1. Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "2. Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "3. Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "4. Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "5. Verifying installation..."
pip list | grep -E "(numpy|pandas|streamlit|plotly|pyyaml|pytest)"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run SEKHMET Recovery Predictor (ACTIVE app):"
echo "  cd src/predictor && streamlit run app.py"
echo ""
echo "To run JMES Synthetic Data Generator (old app):"
echo "  cd src && streamlit run app.py"
echo ""

