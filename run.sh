#!/bin/bash
set -euo pipefail

echo "=== Medicaid Fraud Signal Detection Engine ==="
python3 run.py --output fraud_signals.json "$@"
echo "=== Done. Output: fraud_signals.json ==="
