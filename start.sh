#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "===== OATFLAKE Setup and Startup Script ====="
echo

# Make this script executable (in case it wasn't already)
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
  chmod +x "$(basename "$0")"
fi

# Check if Python 3.10+ is installed
if ! python3 --version | grep -q "3.1[0-9]"; then
  echo "Python 3.10 or higher is required. Please install it and try again."
  exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating a virtual environment..."
  python3 -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if Poetry is installed (within the virtual environment)
if ! command -v poetry &> /dev/null; then
  echo "Poetry is not installed. Installing Poetry..."
  
  # Try curl method first
  if command -v curl &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
  else
    # Fallback to pip if curl is not available
    echo "Curl not found, trying pip installation..."
    pip install poetry
  fi
  
  # Add Poetry to PATH
  export PATH="$HOME/.local/bin:$PATH"
fi

# Configure Poetry to use the project's virtual environment
poetry config virtualenvs.in-project true

# Install dependencies using Poetry
echo "Installing dependencies with Poetry..."
poetry install

# Ensure essential packages are installed
echo "Verifying essential packages..."
pip install --quiet slack-bolt Jinja2

# Success message
echo "Setup complete! Starting the application..."

# Run the application
python run.py

# Keep the window open if there was an error
echo
echo "Application closed. Press Enter to exit..."
read
