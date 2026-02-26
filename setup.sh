#!/bin/bash
set -euo pipefail

echo "=== Medicaid Fraud Detector Setup ==="

# Check Python version
python3 --version | grep -E "3\.(11|12|13)" || {
    echo "ERROR: Python 3.11+ required"
    exit 1
}

# Install dependencies
echo "[1/2] Installing Python dependencies..."
pip install -r requirements.txt

# Download data
echo "[2/2] Downloading data files..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from ingest import download_all
download_all()
"

echo "=== Setup complete ==="
