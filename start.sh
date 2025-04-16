#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if Python 3.10+ is installed
if ! python3 --version | grep -q "3.1[0-9]"; then
  echo "Python 3.10 or higher is required. Please install it and try again."
  exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
  echo "Poetry is not installed. Installing Poetry..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create a virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating a virtual environment..."
  python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies using Poetry
echo "Installing dependencies with Poetry..."
poetry install

# Reinstall all dependencies to ensure they are properly installed
echo "Reinstalling all dependencies with Poetry..."
poetry install --no-root

# Verify slack-bolt installation
echo "Verifying slack-bolt installation..."
if ! pip show slack-bolt &> /dev/null; then
  echo "slack-bolt is not installed. Installing it manually..."
  pip install slack-bolt
fi

# Ensure Jinja2 is installed
if ! pip show Jinja2 &> /dev/null; then
  echo "Jinja2 is not installed. Installing it manually..."
  pip install Jinja2
fi

# Success message
echo "Setup complete! You can now run the application."

# Run the application after setup
echo "Starting the application..."
python run.py
