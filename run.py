"""
run.py - Main orchestrator for Medicaid Provider Fraud Signal Detection
Usage: python run.py [--no-gpu] [--sample N] [--output PATH]
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import download_all, load_leie, load_nppes, load_medicaid_chunks
from signals import (
    signal1_excluded_provider,
    signal2_billing_outlier,
    signal3_rapid_escalation,
    signal4_workforce_impossibility,
    signal5_shared_official,
    signal6_geographic_implausibility,
)
from output import merge_signals, enrich_totals, build_report, save_report


def parse_args():
    p = argparse.ArgumentParser(description="Medicaid Provider Fraud Signal Detection Engine")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU-only mode")
    p.add_argument("--sample", type=int, default=0, help="Process only N rows (for testing)")
    p.add_argument("--output", default="fraud_signals.json", help="Output file path")
    p.add_argument("--data-dir", default="./data", help="Directory for data files")
    p.add_argument("--skip-download", action="store_true", help="Skip downloading data files")
    return p.parse_args()


def main():
    args = parse_args()
    import os
    os.environ["DATA_DIR"] = args.data_dir

    t0 = time.time()
    print("=" * 60)
    print("Medicaid Provider Fraud Signal Detection Engine v1.0.0")
    print("=" * 60)

    # Step 1: Download data
    if not args.skip_download:
        print("\n[STEP 1] Downloading data files...")
        download_all()
    else:
        print("\n[STEP 1] Skipping download (--skip-download)")

    # Step 2: Load reference tables
    print("\n[STEP 2] Loading reference tables...")
    leie = load_leie()
    nppes = load_nppes()

    # Count unique billing NPIs for total_providers_scanned
    print("\n[STEP 3] Counting unique providers...")
    unique_npis = set()
    for chunk in load_medicaid_chunks():
        unique_npis.update(chunk["BILLING_PROVIDER_NPI_NUM"].dropna().unique())
        if args.sample and len(unique_npis) >= args.sample:
            break
    total_scanned = len(unique_npis)
    print(f"[INFO] Total unique billing providers: {total_scanned:,}")

    # Helper to re-iterate chunks
    def chunks():
        count = 0
        for chunk in load_medicaid_chunks():
            yield chunk
            count += len(chunk)
            if args.sample and count >= args.sample * 100:
                break

    # Step 4: Run all 6 signals
    print("\n[STEP 4] Running fraud signal detection...")

    s1 = signal1_excluded_provider(chunks(), leie)
    s2 = signal2_billing_outlier(chunks(), nppes)
    s3 = signal3_rapid_escalation(chunks(), nppes)
    s4 = signal4_workforce_impossibility(chunks(), nppes)
    s5 = signal5_shared_official(chunks(), nppes)
    s6 = signal6_geographic_implausibility(chunks(), nppes)

    # Step 5: Merge and enrich
    print("\n[STEP 5] Merging results...")
    merged = merge_signals(nppes, s1, s2, s3, s4, s5, s6)
    merged = enrich_totals(merged, chunks)

    # Step 6: Build and save report
    print("\n[STEP 6] Generating report...")
    report = build_report(merged, total_scanned)
    save_report(report, args.output)

    elapsed = time.time() - t0
    print(f"\n[DONE] Total runtime: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
