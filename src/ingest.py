"""
ingest.py - Data loading and joining for Medicaid Fraud Detector
Optimized for low memory (12-32GB RAM) using chunked processing
"""

import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

MEDICAID_URL = "https://stopendataprod.blob.core.windows.net/datasets/medicaid-provider-spending/2026-02-09/medicaid-provider-spending.parquet"
LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
NPPES_URL = "https://download.cms.gov/nppes/NPPES_Data_Dissemination_February_2026_V2.zip"

MEDICAID_PATH = DATA_DIR / "medicaid-provider-spending.parquet"
LEIE_PATH = DATA_DIR / "LEIE.csv"
NPPES_PATH = DATA_DIR / "NPPES.csv"


def download_file(url: str, dest: Path, desc: str):
    """Download file with progress bar."""
    if dest.exists():
        print(f"[SKIP] {desc} already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {desc} from {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[OK] {desc} saved to {dest}")


def download_nppes():
    """Download and extract NPPES zip."""
    if NPPES_PATH.exists():
        print(f"[SKIP] NPPES already extracted: {NPPES_PATH}")
        return
    zip_path = DATA_DIR / "nppes.zip"
    download_file(NPPES_URL, zip_path, "NPPES NPI Registry")
    print("[EXTRACT] Extracting NPPES zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find the main CSV (largest file)
        csv_files = [f for f in z.namelist() if f.endswith(".csv") and "FileHeader" not in f]
        main_csv = max(csv_files, key=lambda f: z.getinfo(f).file_size)
        print(f"[EXTRACT] Extracting {main_csv}...")
        with z.open(main_csv) as src, open(NPPES_PATH, "wb") as dst:
            dst.write(src.read())
    print(f"[OK] NPPES extracted to {NPPES_PATH}")


def download_all():
    """Download all required data files."""
    download_file(MEDICAID_URL, MEDICAID_PATH, "Medicaid Provider Spending")
    download_file(LEIE_URL, LEIE_PATH, "OIG LEIE Exclusion List")
    download_nppes()


def load_leie() -> pd.DataFrame:
    """Load and clean the LEIE exclusion list."""
    print("[LOAD] Loading LEIE exclusion list...")
    df = pd.read_csv(LEIE_PATH, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    # Normalize NPI - strip whitespace, keep only 10-digit strings
    df["NPI"] = df["NPI"].str.strip()
    df["NPI"] = df["NPI"].where(df["NPI"].str.match(r"^\d{10}$", na=False))
    # Parse dates - format YYYYMMDD
    for col in ["EXCLDATE", "REINDATE"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    print(f"[OK] LEIE loaded: {len(df):,} rows, {df['NPI'].notna().sum():,} with NPI")
    return df


def load_nppes() -> pd.DataFrame:
    """Load required NPPES columns only."""
    print("[LOAD] Loading NPPES NPI registry (selected columns)...")
    cols = [
        "NPI",
        "Entity Type Code",
        "Provider Organization Name (Legal Business Name)",
        "Provider Last Name (Legal Name)",
        "Provider First Name",
        "Provider Business Practice Location Address State Name",
        "Provider Business Practice Location Address Postal Code",
        "Healthcare Provider Taxonomy Code_1",
        "Provider Enumeration Date",
        "Authorized Official Last Name",
        "Authorized Official First Name",
    ]
    # Read in chunks to save memory
    chunks = []
    chunk_iter = pd.read_csv(
        NPPES_PATH,
        usecols=lambda c: c in cols,
        dtype=str,
        chunksize=500_000,
        low_memory=False,
    )
    for chunk in tqdm(chunk_iter, desc="NPPES chunks"):
        chunk["NPI"] = chunk["NPI"].str.strip()
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["Provider Enumeration Date"] = pd.to_datetime(
        df["Provider Enumeration Date"], format="%m/%d/%Y", errors="coerce"
    )
    # Build provider name
    df["provider_name"] = df.apply(
        lambda r: r["Provider Organization Name (Legal Business Name)"]
        if r["Entity Type Code"] == "2"
        else f"{r.get('Provider First Name', '')} {r.get('Provider Last Name (Legal Name)', '')}".strip(),
        axis=1,
    )
    print(f"[OK] NPPES loaded: {len(df):,} rows")
    return df


def load_medicaid_chunks(chunksize: int = 2_000_000):
    """Yield chunks of the Medicaid parquet file."""
    print(f"[LOAD] Reading Medicaid parquet in chunks of {chunksize:,}...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(MEDICAID_PATH)
    total_rows = pf.metadata.num_rows
    print(f"[INFO] Total rows: {total_rows:,}")
    for batch in pf.iter_batches(batch_size=chunksize):
        df = batch.to_pandas()
        df["CLAIM_FROM_MONTH"] = pd.to_datetime(df["CLAIM_FROM_MONTH"], errors="coerce")
        yield df


def load_medicaid_full() -> pd.DataFrame:
    """Load full Medicaid dataset - requires ~20GB RAM."""
    print("[LOAD] Loading full Medicaid dataset...")
    import pyarrow.parquet as pq
    df = pq.read_table(MEDICAID_PATH).to_pandas()
    df["CLAIM_FROM_MONTH"] = pd.to_datetime(df["CLAIM_FROM_MONTH"], errors="coerce")
    print(f"[OK] Medicaid loaded: {len(df):,} rows")
    return df
