"""
output.py - JSON report generation for Medicaid Fraud Detector
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "1.0.0"


def merge_signals(nppes, *signal_dicts) -> dict:
    """Merge all signal results into a unified provider dict."""
    merged = {}

    for signal_dict in signal_dicts:
        for key, sig in signal_dict.items():
            npi = sig["npi"]
            if npi not in merged:
                # Look up provider info from NPPES
                nppes_row = nppes[nppes["NPI"] == npi]
                if not nppes_row.empty:
                    row = nppes_row.iloc[0]
                    entity_type = "organization" if row.get("Entity Type Code") == "2" else "individual"
                    provider_name = row.get("provider_name", "UNKNOWN")
                    taxonomy = row.get("Healthcare Provider Taxonomy Code_1", "")
                    state = row.get("Provider Business Practice Location Address State Name", "")
                    enum_date = str(row.get("Provider Enumeration Date", ""))[:10]
                else:
                    entity_type = "unknown"
                    provider_name = "UNKNOWN"
                    taxonomy = ""
                    state = ""
                    enum_date = ""

                merged[npi] = {
                    "npi": npi,
                    "provider_name": provider_name,
                    "entity_type": entity_type,
                    "taxonomy_code": taxonomy,
                    "state": state,
                    "enumeration_date": enum_date,
                    "total_paid_all_time": 0.0,
                    "total_claims_all_time": 0,
                    "total_unique_beneficiaries_all_time": 0,
                    "signals": [],
                    "estimated_overpayment_usd": 0.0,
                }

            # Add this signal
            merged[npi]["signals"].append({
                "signal_type": sig["signal_type"],
                "severity": sig["severity"],
                "evidence": sig["evidence"],
                "fca_relevance": sig["fca_relevance"],
            })
            merged[npi]["estimated_overpayment_usd"] += sig.get("estimated_overpayment_usd", 0.0)

    return merged


def enrich_totals(merged: dict, medicaid_chunks_fn) -> dict:
    """Add total_paid, total_claims, total_beneficiaries for each flagged NPI."""
    import pandas as pd

    flagged_npis = set(merged.keys())
    totals = {npi: {"paid": 0.0, "claims": 0, "benes": 0} for npi in flagged_npis}

    for chunk in medicaid_chunks_fn():
        sub = chunk[chunk["BILLING_PROVIDER_NPI_NUM"].isin(flagged_npis)]
        for npi, grp in sub.groupby("BILLING_PROVIDER_NPI_NUM"):
            totals[npi]["paid"] += grp["TOTAL_PAID"].sum()
            totals[npi]["claims"] += grp["TOTAL_CLAIMS"].sum()
            totals[npi]["benes"] += grp["TOTAL_UNIQUE_BENEFICIARIES"].sum()

    for npi in merged:
        if npi in totals:
            merged[npi]["total_paid_all_time"] = round(totals[npi]["paid"], 2)
            merged[npi]["total_claims_all_time"] = int(totals[npi]["claims"])
            merged[npi]["total_unique_beneficiaries_all_time"] = int(totals[npi]["benes"])

    return merged


def build_report(merged: dict, total_scanned: int) -> dict:
    """Build the final JSON report structure."""
    providers = list(merged.values())

    # Count signals
    signal_counts = {
        "excluded_provider": 0,
        "billing_outlier": 0,
        "rapid_escalation": 0,
        "workforce_impossibility": 0,
        "shared_official": 0,
        "geographic_implausibility": 0,
    }
    for p in providers:
        for sig in p["signals"]:
            st = sig["signal_type"]
            if st in signal_counts:
                signal_counts[st] += 1

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool_version": TOOL_VERSION,
        "total_providers_scanned": total_scanned,
        "total_providers_flagged": len(providers),
        "signal_counts": signal_counts,
        "flagged_providers": providers,
    }


def save_report(report: dict, output_path: str = "fraud_signals.json"):
    """Save the report to a JSON file."""
    path = Path(output_path)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[OK] Report saved to {path} ({size_mb:.1f} MB)")
    print(f"     Providers scanned: {report['total_providers_scanned']:,}")
    print(f"     Providers flagged: {report['total_providers_flagged']:,}")
    for k, v in report["signal_counts"].items():
        print(f"     {k}: {v}")
