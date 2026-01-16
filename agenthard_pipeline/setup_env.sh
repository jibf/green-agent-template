#!/bin/bash
# Setup script for AgentHard Pipeline conda environment

set -e  # Exit on error

ENV_NAME="agenthard_pipeline"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "AgentHard Pipeline Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborted. Please use a different environment name or remove manually."
        exit 1
    fi
fi

# Create conda environment
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install pip packages
echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Before running the pipeline:"
echo "  1. Copy .env.example to .env"
echo "  2. Fill in your API_KEY and BASE_URL in .env"
echo ""
