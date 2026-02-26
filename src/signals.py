"""
signals.py - All 6 fraud signal implementations for Medicaid Provider Fraud Detector
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Iterator


# ─────────────────────────────────────────────
# Signal 1: Excluded Provider Still Billing
# ─────────────────────────────────────────────

def signal1_excluded_provider(medicaid_chunks: Iterator[pd.DataFrame], leie: pd.DataFrame) -> dict:
    """
    Flag providers whose NPI appears in LEIE exclusion list and
    billed AFTER their exclusion date while not yet reinstated.
    """
    print("[SIGNAL 1] Excluded provider still billing...")

    # Build lookup of excluded NPIs
    leie_npi = leie[leie["NPI"].notna()][["NPI", "EXCLDATE", "REINDATE", "EXCLTYPE"]].copy()
    leie_npi = leie_npi.set_index("NPI")

    results = {}  # npi -> {excl_date, reindate, excltype, total_paid_after, months}

    for chunk in medicaid_chunks:
        for npi_col in ["BILLING_PROVIDER_NPI_NUM", "SERVICING_PROVIDER_NPI_NUM"]:
            sub = chunk[[npi_col, "CLAIM_FROM_MONTH", "TOTAL_PAID"]].copy()
            sub.columns = ["NPI", "CLAIM_FROM_MONTH", "TOTAL_PAID"]
            sub = sub[sub["NPI"].isin(leie_npi.index)]
            if sub.empty:
                continue
            sub = sub.join(leie_npi, on="NPI")
            # Flag: claimed after exclusion AND (no reinstatement OR reinstated after claim)
            mask = (sub["CLAIM_FROM_MONTH"] >= sub["EXCLDATE"]) & (
                sub["REINDATE"].isna() | (sub["CLAIM_FROM_MONTH"] < sub["REINDATE"])
            )
            flagged = sub[mask]
            for npi, grp in flagged.groupby("NPI"):
                if npi not in results:
                    row = leie_npi.loc[npi]
                    results[npi] = {
                        "excl_date": row["EXCLDATE"],
                        "reindate": row["REINDATE"],
                        "excltype": row["EXCLTYPE"],
                        "total_paid_after_excl": 0.0,
                    }
                results[npi]["total_paid_after_excl"] += grp["TOTAL_PAID"].sum()

    flagged_list = []
    for npi, data in results.items():
        flagged_list.append({
            "npi": npi,
            "signal_type": "excluded_provider",
            "severity": "critical",
            "estimated_overpayment_usd": round(data["total_paid_after_excl"], 2),
            "evidence": {
                "exclusion_date": str(data["excl_date"])[:10] if pd.notna(data["excl_date"]) else None,
                "reinstatement_date": str(data["reindate"])[:10] if pd.notna(data["reindate"]) else None,
                "exclusion_type": data["excltype"],
                "total_paid_after_exclusion": round(data["total_paid_after_excl"], 2),
            },
            "fca_relevance": {
                "claim_type": "Excluded provider submitted claims to Medicaid after exclusion date",
                "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
                "suggested_next_steps": [
                    f"Verify exclusion status at OIG LEIE: https://exclusions.oig.hhs.gov/",
                    f"Request all claims submitted by NPI {npi} after exclusion date",
                    "Refer to CMS for overpayment demand and potential False Claims Act action",
                ],
            },
        })

    print(f"[SIGNAL 1] Found {len(flagged_list)} excluded providers still billing")
    return {npi["npi"]: npi for npi in flagged_list}


# ─────────────────────────────────────────────
# Signal 2: Billing Volume Outlier
# ─────────────────────────────────────────────

def signal2_billing_outlier(medicaid_chunks: Iterator[pd.DataFrame], nppes: pd.DataFrame) -> dict:
    """
    Flag providers above 99th percentile of total paid within their taxonomy+state peer group.
    """
    print("[SIGNAL 2] Billing volume outlier...")

    # Aggregate total paid per billing NPI
    totals = {}
    for chunk in medicaid_chunks:
        for _, row in chunk[["BILLING_PROVIDER_NPI_NUM", "TOTAL_PAID"]].iterrows():
            npi = row["BILLING_PROVIDER_NPI_NUM"]
            totals[npi] = totals.get(npi, 0.0) + row["TOTAL_PAID"]

    df = pd.DataFrame(list(totals.items()), columns=["NPI", "total_paid"])

    # Join NPPES for taxonomy and state
    nppes_sub = nppes[["NPI", "Healthcare Provider Taxonomy Code_1",
                        "Provider Business Practice Location Address State Name"]].copy()
    nppes_sub.columns = ["NPI", "taxonomy", "state"]
    df = df.merge(nppes_sub, on="NPI", how="left")
    df["taxonomy"] = df["taxonomy"].fillna("UNKNOWN")
    df["state"] = df["state"].fillna("UNKNOWN")

    # Compute peer group stats
    peer = df.groupby(["taxonomy", "state"])["total_paid"].agg(
        peer_median="median", peer_p99=lambda x: np.percentile(x, 99)
    ).reset_index()
    df = df.merge(peer, on=["taxonomy", "state"], how="left")
    df["ratio"] = df["total_paid"] / df["peer_median"].replace(0, np.nan)

    # Flag above 99th percentile
    flagged = df[df["total_paid"] > df["peer_p99"]].copy()

    results = {}
    for _, row in flagged.iterrows():
        npi = row["NPI"]
        overpay = max(0.0, row["total_paid"] - row["peer_p99"])
        severity = "high" if row.get("ratio", 0) > 5 else "medium"
        results[npi] = {
            "npi": npi,
            "signal_type": "billing_outlier",
            "severity": severity,
            "estimated_overpayment_usd": round(overpay, 2),
            "evidence": {
                "total_paid": round(row["total_paid"], 2),
                "peer_median": round(row["peer_median"], 2),
                "peer_p99": round(row["peer_p99"], 2),
                "ratio_vs_median": round(row.get("ratio", 0), 2),
                "taxonomy_code": row["taxonomy"],
                "state": row["state"],
            },
            "fca_relevance": {
                "claim_type": "Provider billing volume exceeds 99th percentile of peer group, indicating potential overbilling",
                "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
                "suggested_next_steps": [
                    f"Request itemized claims detail for NPI {npi} in taxonomy {row['taxonomy']}",
                    f"Compare procedure codes billed against credentialing records for NPI {npi}",
                    "Engage clinical reviewer to assess medical necessity of high-volume procedures",
                ],
            },
        }

    print(f"[SIGNAL 2] Found {len(results)} billing outliers")
    return results


# ─────────────────────────────────────────────
# Signal 3: Rapid Billing Escalation (New Entity)
# ─────────────────────────────────────────────

def signal3_rapid_escalation(medicaid_chunks: Iterator[pd.DataFrame], nppes: pd.DataFrame) -> dict:
    """
    Flag new providers (enumerated <24 months before first billing) with >200% rolling 3-month growth.
    """
    print("[SIGNAL 3] Rapid billing escalation (new entities)...")

    # Aggregate monthly paid per NPI
    monthly = {}
    for chunk in medicaid_chunks:
        for _, row in chunk[["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH", "TOTAL_PAID"]].iterrows():
            npi = row["BILLING_PROVIDER_NPI_NUM"]
            month = row["CLAIM_FROM_MONTH"]
            if npi not in monthly:
                monthly[npi] = {}
            monthly[npi][month] = monthly[npi].get(month, 0.0) + row["TOTAL_PAID"]

    # Get enumeration dates
    enum_dates = nppes[["NPI", "Provider Enumeration Date"]].set_index("NPI")["Provider Enumeration Date"]

    results = {}
    for npi, month_data in monthly.items():
        if npi not in enum_dates.index:
            continue
        enum_date = enum_dates[npi]
        if pd.isna(enum_date):
            continue

        months_sorted = sorted(month_data.keys())
        first_billing = months_sorted[0]

        # Only flag new entities: enumerated within 24 months before first billing
        months_gap = (first_billing - enum_date).days / 30
        if not (0 <= months_gap <= 24):
            continue

        # Get first 12 months of billing
        first_12 = months_sorted[:12]
        paid_series = [month_data[m] for m in first_12]

        if len(paid_series) < 3:
            continue

        # Compute rolling 3-month average growth rate
        peak_growth = 0.0
        for i in range(2, len(paid_series)):
            window = paid_series[i-2:i+1]
            if window[0] > 0:
                growth = (window[2] - window[0]) / window[0] * 100
                peak_growth = max(peak_growth, growth)

        if peak_growth > 200:
            severity = "high" if peak_growth > 500 else "medium"
            flagged_months_paid = sum(
                paid_series[i] for i in range(2, len(paid_series))
                if paid_series[i-2] > 0 and (paid_series[i] - paid_series[i-2]) / paid_series[i-2] * 100 > 200
            )
            results[npi] = {
                "npi": npi,
                "signal_type": "rapid_escalation",
                "severity": severity,
                "estimated_overpayment_usd": round(flagged_months_paid, 2),
                "evidence": {
                    "enumeration_date": str(enum_date)[:10],
                    "first_billing_month": str(first_billing)[:7],
                    "months_to_first_billing": round(months_gap, 1),
                    "monthly_paid_first_12": {str(m)[:7]: round(v, 2) for m, v in zip(first_12, paid_series)},
                    "peak_3month_growth_rate_pct": round(peak_growth, 1),
                },
                "fca_relevance": {
                    "claim_type": "New provider shows explosive billing growth indicative of bust-out fraud scheme",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(A)",
                    "suggested_next_steps": [
                        f"Verify NPI {npi} physical location is operational via site visit",
                        f"Request patient records for claims submitted during peak growth months",
                        "Check if authorized official has prior fraud history in OIG LEIE",
                    ],
                },
            }

    print(f"[SIGNAL 3] Found {len(results)} rapid escalation cases")
    return results


# ─────────────────────────────────────────────
# Signal 4: Workforce Impossibility
# ─────────────────────────────────────────────

def signal4_workforce_impossibility(medicaid_chunks: Iterator[pd.DataFrame], nppes: pd.DataFrame) -> dict:
    """
    Flag organizations where implied claims-per-hour exceeds 6 in any single month.
    Threshold: total_claims / 22 days / 8 hours > 6
    """
    print("[SIGNAL 4] Workforce impossibility...")

    # Get organization NPIs
    org_npis = set(nppes[nppes["Entity Type Code"] == "2"]["NPI"].tolist())

    # Find peak monthly claims per organization NPI
    peak = {}  # npi -> {month, claims, paid}
    for chunk in medicaid_chunks:
        orgs = chunk[chunk["BILLING_PROVIDER_NPI_NUM"].isin(org_npis)]
        monthly = orgs.groupby(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"]).agg(
            total_claims=("TOTAL_CLAIMS", "sum"),
            total_paid=("TOTAL_PAID", "sum"),
        ).reset_index()
        for _, row in monthly.iterrows():
            npi = row["BILLING_PROVIDER_NPI_NUM"]
            if npi not in peak or row["total_claims"] > peak[npi]["claims"]:
                peak[npi] = {
                    "month": row["CLAIM_FROM_MONTH"],
                    "claims": row["total_claims"],
                    "paid": row["total_paid"],
                }

    THRESHOLD = 6 * 8 * 22  # 1,056 claims/month max

    results = {}
    for npi, data in peak.items():
        implied_per_hour = data["claims"] / 22 / 8
        if implied_per_hour > 6:
            results[npi] = {
                "npi": npi,
                "signal_type": "workforce_impossibility",
                "severity": "high",
                "estimated_overpayment_usd": round(
                    max(0, (data["claims"] - THRESHOLD)) * (data["paid"] / data["claims"]) if data["claims"] > 0 else 0,
                    2,
                ),
                "evidence": {
                    "peak_month": str(data["month"])[:7],
                    "peak_claims_count": int(data["claims"]),
                    "implied_claims_per_hour": round(implied_per_hour, 2),
                    "total_paid_peak_month": round(data["paid"], 2),
                    "threshold_claims_per_hour": 6,
                },
                "fca_relevance": {
                    "claim_type": "Organization billing volume physically impossible — implies fabricated claims",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(B)",
                    "suggested_next_steps": [
                        f"Request staffing records for NPI {npi} during peak month {str(data['month'])[:7]}",
                        f"Audit patient records for claims during peak billing month to verify encounters occurred",
                        "Cross-reference provider enrollment with number of licensed practitioners on staff",
                    ],
                },
            }

    print(f"[SIGNAL 4] Found {len(results)} workforce impossibility cases")
    return results


# ─────────────────────────────────────────────
# Signal 5: Shared Authorized Official
# ─────────────────────────────────────────────

def signal5_shared_official(medicaid_chunks: Iterator[pd.DataFrame], nppes: pd.DataFrame) -> dict:
    """
    Flag authorized officials controlling 5+ NPIs with combined billing > $1M.
    """
    print("[SIGNAL 5] Shared authorized official...")

    # Group NPIs by authorized official
    officials = nppes[
        nppes["Authorized Official Last Name"].notna() &
        nppes["Authorized Official First Name"].notna()
    ].copy()
    officials["official_key"] = (
        officials["Authorized Official Last Name"].str.upper().str.strip() + "|" +
        officials["Authorized Official First Name"].str.upper().str.strip()
    )
    official_groups = officials.groupby("official_key")["NPI"].apply(list)
    # Only keep officials with 5+ NPIs
    official_groups = official_groups[official_groups.apply(len) >= 5]

    if official_groups.empty:
        print("[SIGNAL 5] No shared officials with 5+ NPIs found")
        return {}

    all_npis = set(npi for npis in official_groups for npi in npis)

    # Aggregate paid per NPI
    npi_totals = {}
    for chunk in medicaid_chunks:
        sub = chunk[chunk["BILLING_PROVIDER_NPI_NUM"].isin(all_npis)]
        for npi, grp in sub.groupby("BILLING_PROVIDER_NPI_NUM"):
            npi_totals[npi] = npi_totals.get(npi, 0.0) + grp["TOTAL_PAID"].sum()

    results = {}
    for official_key, npis in official_groups.items():
        combined = sum(npi_totals.get(npi, 0.0) for npi in npis)
        if combined > 1_000_000:
            last, first = official_key.split("|")
            severity = "high" if combined > 5_000_000 else "medium"
            results[official_key] = {
                "npi": npis[0],  # primary NPI for record
                "signal_type": "shared_official",
                "severity": severity,
                "estimated_overpayment_usd": 0.0,  # per spec
                "evidence": {
                    "authorized_official": f"{first} {last}",
                    "controlled_npis": npis,
                    "npi_count": len(npis),
                    "paid_per_npi": {npi: round(npi_totals.get(npi, 0.0), 2) for npi in npis},
                    "combined_total_paid": round(combined, 2),
                },
                "fca_relevance": {
                    "claim_type": "Single individual controls multiple billing entities — potential shell entity conspiracy",
                    "statute_reference": "31 U.S.C. section 3729(a)(1)(C)",
                    "suggested_next_steps": [
                        f"Investigate corporate ownership of all NPIs controlled by {first} {last}",
                        f"Check if any of the {len(npis)} controlled NPIs share addresses or phone numbers",
                        "Review LEIE for prior exclusions of the authorized official by name",
                    ],
                },
            }

    print(f"[SIGNAL 5] Found {len(results)} shared official cases")
    return results


# ─────────────────────────────────────────────
# Signal 6: Geographic Implausibility
# ─────────────────────────────────────────────

HOME_HEALTH_CODES = set([
    "G0151", "G0152", "G0153", "G0154", "G0155", "G0156", "G0157", "G0158",
    "G0159", "G0160", "G0161", "G0162",
    "G0299", "G0300",
    "S9122", "S9123", "S9124",
    "T1019", "T1020", "T1021", "T1022",
])


def signal6_geographic_implausibility(medicaid_chunks: Iterator[pd.DataFrame], nppes: pd.DataFrame) -> dict:
    """
    Flag providers billing home health codes with unique_beneficiaries/claims ratio < 0.1.
    """
    print("[SIGNAL 6] Geographic implausibility (home health)...")

    npi_state = nppes[["NPI", "Provider Business Practice Location Address State Name"]].set_index("NPI")[
        "Provider Business Practice Location Address State Name"
    ]

    results = {}
    for chunk in medicaid_chunks:
        hh = chunk[chunk["HCPCS_CODE"].isin(HOME_HEALTH_CODES)].copy()
        if hh.empty:
            continue
        monthly = hh.groupby(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH", "HCPCS_CODE"]).agg(
            total_claims=("TOTAL_CLAIMS", "sum"),
            total_beneficiaries=("TOTAL_UNIQUE_BENEFICIARIES", "sum"),
        ).reset_index()
        # Only rows with >100 claims in that month
        monthly = monthly[monthly["total_claims"] > 100]
        monthly["ratio"] = monthly["total_beneficiaries"] / monthly["total_claims"]
        flagged = monthly[monthly["ratio"] < 0.1]
        for _, row in flagged.iterrows():
            npi = row["BILLING_PROVIDER_NPI_NUM"]
            key = f"{npi}_{row['CLAIM_FROM_MONTH']}_{row['HCPCS_CODE']}"
            state = npi_state.get(npi, "UNKNOWN")
            if npi not in results:
                results[npi] = {
                    "npi": npi,
                    "signal_type": "geographic_implausibility",
                    "severity": "medium",
                    "estimated_overpayment_usd": 0.0,  # per spec
                    "evidence": {
                        "state": state,
                        "flagged_instances": [],
                    },
                    "fca_relevance": {
                        "claim_type": "Home health provider billing same patients repeatedly — reverse false claims pattern",
                        "statute_reference": "31 U.S.C. section 3729(a)(1)(G)",
                        "suggested_next_steps": [
                            f"Request home health visit records for NPI {npi} for flagged months",
                            f"Verify patient consent and physician orders for repeated home health visits",
                            "Cross-reference beneficiary IDs to confirm distinct patient encounters",
                        ],
                    },
                }
            results[npi]["evidence"]["flagged_instances"].append({
                "hcpcs_code": row["HCPCS_CODE"],
                "month": str(row["CLAIM_FROM_MONTH"])[:7],
                "claims_count": int(row["total_claims"]),
                "unique_beneficiaries": int(row["total_beneficiaries"]),
                "ratio": round(row["ratio"], 4),
            })

    print(f"[SIGNAL 6] Found {len(results)} geographic implausibility cases")
    return results
