# Medicaid Provider Fraud Signal Detection Engine

A CLI tool that ingests the HHS Medicaid Provider Spending dataset and detects 6 fraud signals using cross-referencing with OIG LEIE and NPPES NPI registry.

## Requirements

- Python 3.11+
- 16GB+ RAM (32GB recommended for full run)
- ~10GB free disk space
- macOS 14+ or Ubuntu 22.04+

## Quick Start

```bash
# 1. Install dependencies and download data
bash setup.sh

# 2. Run full detection
bash run.sh

# Output: fraud_signals.json
```

## Manual Run

```bash
pip install -r requirements.txt
python run.py --output fraud_signals.json
```

## Options

```
--no-gpu         Force CPU-only mode (default: auto-detect)
--sample N       Process only N rows (useful for testing)
--output PATH    Output file path (default: fraud_signals.json)
--data-dir PATH  Directory for data files (default: ./data)
--skip-download  Skip downloading data (if already downloaded)
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Fraud Signals Implemented

| Signal | Description | FCA Statute |
|--------|-------------|-------------|
| 1 | Excluded provider still billing after LEIE exclusion date | 3729(a)(1)(A) |
| 2 | Billing volume outlier — above 99th percentile of peer group | 3729(a)(1)(A) |
| 3 | Rapid billing escalation in new entities | 3729(a)(1)(A) |
| 4 | Workforce impossibility — claims-per-hour exceeds 6 | 3729(a)(1)(B) |
| 5 | Shared authorized official across 5+ NPIs with >$1M combined | 3729(a)(1)(C) |
| 6 | Geographic implausibility in home health billing | 3729(a)(1)(G) |

## File Structure

```
submission/
  README.md           This file
  requirements.txt    Python dependencies
  setup.sh            Download data + install deps
  run.sh              Run full detection
  run.py              Main orchestrator
  src/
    ingest.py         Data loading (chunked, low-memory)
    signals.py        All 6 signal implementations
    output.py         JSON report generation
  tests/
    test_signals.py   Unit tests (one per signal)
    fixtures/         Synthetic test data
  fraud_signals.json  Sample output
```

## Performance

| Environment | Expected Runtime |
|-------------|-----------------|
| Linux 200GB RAM + GPU | < 30 min |
| Linux 64GB RAM, no GPU | < 60 min |
| MacBook 32GB Apple Silicon | ~2 hours |
| MacBook 16GB Apple Silicon | ~4 hours |
