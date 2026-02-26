"""
tests/test_signals.py - Unit tests for all 6 fraud signals using synthetic fixtures
Run with: pytest tests/ -v
"""

import sys
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import (
    signal1_excluded_provider,
    signal2_billing_outlier,
    signal3_rapid_escalation,
    signal4_workforce_impossibility,
    signal5_shared_official,
    signal6_geographic_implausibility,
)


# ─── Fixtures ───────────────────────────────

def make_medicaid_chunk(rows):
    """Create a synthetic Medicaid dataframe."""
    df = pd.DataFrame(rows)
    df["CLAIM_FROM_MONTH"] = pd.to_datetime(df["CLAIM_FROM_MONTH"])
    return df


def chunks_from(df):
    """Yield a single chunk."""
    yield df


# ─── Signal 1 Tests ─────────────────────────

class TestSignal1ExcludedProvider:
    def test_flags_excluded_provider_billing_after_exclusion(self):
        leie = pd.DataFrame([{
            "NPI": "1234567890",
            "EXCLDATE": pd.Timestamp("2020-01-01"),
            "REINDATE": pd.NaT,
            "EXCLTYPE": "PERMEXCL",
        }])

        medicaid = make_medicaid_chunk([{
            "BILLING_PROVIDER_NPI_NUM": "1234567890",
            "SERVICING_PROVIDER_NPI_NUM": "9999999999",
            "CLAIM_FROM_MONTH": "2021-06-01",
            "TOTAL_PAID": 50000.0,
            "TOTAL_CLAIMS": 100,
            "TOTAL_UNIQUE_BENEFICIARIES": 50,
            "HCPCS_CODE": "99213",
        }])

        result = signal1_excluded_provider(chunks_from(medicaid), leie)
        assert "1234567890" in result
        sig = result["1234567890"]
        assert sig["signal_type"] == "excluded_provider"
        assert sig["severity"] == "critical"
        assert sig["estimated_overpayment_usd"] == 50000.0

    def test_does_not_flag_before_exclusion(self):
        leie = pd.DataFrame([{
            "NPI": "1234567890",
            "EXCLDATE": pd.Timestamp("2022-01-01"),
            "REINDATE": pd.NaT,
            "EXCLTYPE": "PERMEXCL",
        }])

        medicaid = make_medicaid_chunk([{
            "BILLING_PROVIDER_NPI_NUM": "1234567890",
            "SERVICING_PROVIDER_NPI_NUM": "9999999999",
            "CLAIM_FROM_MONTH": "2021-06-01",
            "TOTAL_PAID": 50000.0,
            "TOTAL_CLAIMS": 100,
            "TOTAL_UNIQUE_BENEFICIARIES": 50,
            "HCPCS_CODE": "99213",
        }])

        result = signal1_excluded_provider(chunks_from(medicaid), leie)
        assert "1234567890" not in result

    def test_handles_missing_npi_in_leie(self):
        """LEIE rows without NPI should not crash."""
        leie = pd.DataFrame([{
            "NPI": None,
            "EXCLDATE": pd.Timestamp("2020-01-01"),
            "REINDATE": pd.NaT,
            "EXCLTYPE": "PERMEXCL",
        }])

        medicaid = make_medicaid_chunk([{
            "BILLING_PROVIDER_NPI_NUM": "1234567890",
            "SERVICING_PROVIDER_NPI_NUM": "9999999999",
            "CLAIM_FROM_MONTH": "2021-06-01",
            "TOTAL_PAID": 50000.0,
            "TOTAL_CLAIMS": 100,
            "TOTAL_UNIQUE_BENEFICIARIES": 50,
            "HCPCS_CODE": "99213",
        }])

        result = signal1_excluded_provider(chunks_from(medicaid), leie)
        assert "1234567890" not in result  # No match without NPI


# ─── Signal 2 Tests ─────────────────────────

class TestSignal2BillingOutlier:
    def test_flags_provider_above_99th_percentile(self):
        # Create 100 providers in same taxonomy+state, one is massive outlier
        rows = []
        for i in range(99):
            rows.append({
                "BILLING_PROVIDER_NPI_NUM": f"000000{i:04d}",
                "SERVICING_PROVIDER_NPI_NUM": "9999999999",
                "CLAIM_FROM_MONTH": "2022-01-01",
                "TOTAL_PAID": 10000.0,  # Normal
                "TOTAL_CLAIMS": 50,
                "TOTAL_UNIQUE_BENEFICIARIES": 40,
                "HCPCS_CODE": "99213",
            })
        # Outlier
        rows.append({
            "BILLING_PROVIDER_NPI_NUM": "OUTLIER0001",
            "SERVICING_PROVIDER_NPI_NUM": "9999999999",
            "CLAIM_FROM_MONTH": "2022-01-01",
            "TOTAL_PAID": 5_000_000.0,  # Massive outlier
            "TOTAL_CLAIMS": 50,
            "TOTAL_UNIQUE_BENEFICIARIES": 40,
            "HCPCS_CODE": "99213",
        })

        medicaid = make_medicaid_chunk(rows)

        nppes = pd.DataFrame([{
            "NPI": "OUTLIER0001",
            "Entity Type Code": "1",
            "Healthcare Provider Taxonomy Code_1": "207Q00000X",
            "Provider Business Practice Location Address State Name": "FL",
            "provider_name": "Dr. Outlier",
            "Provider Enumeration Date": pd.Timestamp("2015-01-01"),
            "Authorized Official Last Name": None,
            "Authorized Official First Name": None,
        }] + [{
            "NPI": f"000000{i:04d}",
            "Entity Type Code": "1",
            "Healthcare Provider Taxonomy Code_1": "207Q00000X",
            "Provider Business Practice Location Address State Name": "FL",
            "provider_name": f"Dr. Normal {i}",
            "Provider Enumeration Date": pd.Timestamp("2015-01-01"),
            "Authorized Official Last Name": None,
            "Authorized Official First Name": None,
        } for i in range(99)])

        result = signal2_billing_outlier(chunks_from(medicaid), nppes)
        assert "OUTLIER0001" in result
        assert result["OUTLIER0001"]["signal_type"] == "billing_outlier"


# ─── Signal 3 Tests ─────────────────────────

class TestSignal3RapidEscalation:
    def test_flags_new_provider_with_explosive_growth(self):
        base = datetime(2021, 1, 1)
        rows = []
        # Month 1: $1k, Month 2: $3k, Month 3: $10k (>200% growth)
        paid = [1000, 3000, 10000, 30000, 90000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]
        for i, p in enumerate(paid):
            rows.append({
                "BILLING_PROVIDER_NPI_NUM": "NEWBIZ0001",
                "SERVICING_PROVIDER_NPI_NUM": "9999999999",
                "CLAIM_FROM_MONTH": (base + timedelta(days=30*i)).strftime("%Y-%m-01"),
                "TOTAL_PAID": float(p),
                "TOTAL_CLAIMS": 10,
                "TOTAL_UNIQUE_BENEFICIARIES": 8,
                "HCPCS_CODE": "99213",
            })

        medicaid = make_medicaid_chunk(rows)

        nppes = pd.DataFrame([{
            "NPI": "NEWBIZ0001",
            "Provider Enumeration Date": pd.Timestamp("2020-06-01"),  # 7 months before first billing
        }])

        result = signal3_rapid_escalation(chunks_from(medicaid), nppes)
        assert "NEWBIZ0001" in result
        assert result["NEWBIZ0001"]["signal_type"] == "rapid_escalation"


# ─── Signal 4 Tests ─────────────────────────

class TestSignal4WorkforceImpossibility:
    def test_flags_organization_with_impossible_claims(self):
        # 6 * 8 * 22 = 1056 max; we'll put 10000 claims
        medicaid = make_medicaid_chunk([{
            "BILLING_PROVIDER_NPI_NUM": "ORGBIG0001",
            "SERVICING_PROVIDER_NPI_NUM": "9999999999",
            "CLAIM_FROM_MONTH": "2022-03-01",
            "TOTAL_PAID": 500000.0,
            "TOTAL_CLAIMS": 10000,
            "TOTAL_UNIQUE_BENEFICIARIES": 200,
            "HCPCS_CODE": "99213",
        }])

        nppes = pd.DataFrame([{
            "NPI": "ORGBIG0001",
            "Entity Type Code": "2",  # Organization
            "provider_name": "Big Org LLC",
            "Healthcare Provider Taxonomy Code_1": "261QM1300X",
            "Provider Business Practice Location Address State Name": "TX",
            "Provider Enumeration Date": pd.Timestamp("2015-01-01"),
            "Authorized Official Last Name": "Smith",
            "Authorized Official First Name": "John",
        }])

        result = signal4_workforce_impossibility(chunks_from(medicaid), nppes)
        assert "ORGBIG0001" in result
        sig = result["ORGBIG0001"]
        assert sig["signal_type"] == "workforce_impossibility"
        assert sig["severity"] == "high"
        assert sig["evidence"]["implied_claims_per_hour"] > 6


# ─── Signal 5 Tests ─────────────────────────

class TestSignal5SharedOfficial:
    def test_flags_official_controlling_5_plus_npis_over_1m(self):
        # 6 NPIs controlled by same person, each billing $300k = $1.8M total
        nppes_rows = []
        for i in range(6):
            nppes_rows.append({
                "NPI": f"SHELL{i:06d}",
                "Entity Type Code": "2",
                "Provider Organization Name (Legal Business Name)": f"Shell LLC {i}",
                "Provider Last Name (Legal Name)": None,
                "Provider First Name": None,
                "Provider Business Practice Location Address State Name": "FL",
                "Provider Business Practice Location Address Postal Code": "33101",
                "Healthcare Provider Taxonomy Code_1": "261QM1300X",
                "Provider Enumeration Date": pd.Timestamp("2018-01-01"),
                "Authorized Official Last Name": "JONES",
                "Authorized Official First Name": "MARK",
                "provider_name": f"Shell LLC {i}",
            })

        nppes = pd.DataFrame(nppes_rows)

        medicaid_rows = []
        for i in range(6):
            medicaid_rows.append({
                "BILLING_PROVIDER_NPI_NUM": f"SHELL{i:06d}",
                "SERVICING_PROVIDER_NPI_NUM": "9999999999",
                "CLAIM_FROM_MONTH": "2022-01-01",
                "TOTAL_PAID": 300000.0,
                "TOTAL_CLAIMS": 500,
                "TOTAL_UNIQUE_BENEFICIARIES": 100,
                "HCPCS_CODE": "99213",
            })

        medicaid = make_medicaid_chunk(medicaid_rows)

        result = signal5_shared_official(chunks_from(medicaid), nppes)
        assert len(result) > 0
        key = list(result.keys())[0]
        assert result[key]["signal_type"] == "shared_official"
        assert result[key]["evidence"]["combined_total_paid"] == 1_800_000.0


# ─── Signal 6 Tests ─────────────────────────

class TestSignal6GeographicImplausibility:
    def test_flags_home_health_low_beneficiary_ratio(self):
        # 500 claims, only 10 unique beneficiaries = ratio 0.02 < 0.1
        medicaid = make_medicaid_chunk([{
            "BILLING_PROVIDER_NPI_NUM": "HOMEH0001",
            "SERVICING_PROVIDER_NPI_NUM": "9999999999",
            "CLAIM_FROM_MONTH": "2022-05-01",
            "TOTAL_PAID": 200000.0,
            "TOTAL_CLAIMS": 500,
            "TOTAL_UNIQUE_BENEFICIARIES": 10,
            "HCPCS_CODE": "T1019",  # Home health code
        }])

        nppes = pd.DataFrame([{
            "NPI": "HOMEH0001",
            "Provider Business Practice Location Address State Name": "FL",
        }])

        result = signal6_geographic_implausibility(chunks_from(medicaid), nppes)
        assert "HOMEH0001" in result
        sig = result["HOMEH0001"]
        assert sig["signal_type"] == "geographic_implausibility"
        instance = sig["evidence"]["flagged_instances"][0]
        assert instance["ratio"] < 0.1
