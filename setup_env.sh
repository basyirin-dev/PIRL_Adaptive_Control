#!/bin/zsh

# PIRL Environment Setup Script
# System: OpenSUSE Tumbleweed / ZSH
# Author: FastTrack Supervisor

# 1. Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "❌ CRITICAL: Python 3 could not be found."
    exit 1
fi

echo ">>> Initializing Python Virtual Environment..."

# 2. Create venv (The Clean Room)
# We name it '.venv' so it remains hidden from the file explorer view
python3 -m venv .venv

# 3. Activate the Environment
echo ">>> Activating .venv..."
source .venv/bin/activate

# 4. Upgrade pip (Always do this first)
echo ">>> Upgrading pip..."
pip install --upgrade pip

# 5. Install Dependencies
if [ -f "requirements.txt" ]; then
    echo ">>> Installing Dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "❌ ERROR: requirements.txt not found!"
    exit 1
fi

echo ">>> SUCCESS: Environment is ready."
echo ">>> To activate in the future, run: source .venv/bin/activate"