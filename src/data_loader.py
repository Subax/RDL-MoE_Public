import pandas as pd
import numpy as np
import re
from .config import CLINICAL_PATH, RADIOMICS_PATH


def load_data():
    print(f"Loading Data from:\n - {CLINICAL_PATH}\n - {RADIOMICS_PATH}")
    try:
        df_clin = pd.read_csv(CLINICAL_PATH)
        df_rad = pd.read_csv(RADIOMICS_PATH)
    except FileNotFoundError:
        print(
            "Warning: Data files not found. Please ensure 'data/' folder exists with CSV files."
        )
        raise

    id_col = next(
        (
            c
            for c in df_rad.columns
            if "ANON" in str(c) or df_rad[c].astype(str).str.contains("ANON").any()
        ),
        None,
    )

    df_rad["PatientID"] = df_rad[id_col].apply(
        lambda x: (
            re.search(r"(ANON\d{4})", str(x)).group(1)
            if re.search(r"(ANON\d{4})", str(x))
            else None
        )
    )
    df_rad = df_rad.dropna(subset=["PatientID"])
    df_merged = pd.merge(df_clin, df_rad, on="PatientID", how="inner")

    df_merged["Survival.time"] = pd.to_numeric(
        df_merged["Survival.time"], errors="coerce"
    )
    df_merged["Event"] = (
        df_merged["deadstatus.event"]
        .astype(str)
        .str.strip()
        .isin(["1", "True", "true", "1.0", 1])
        .astype(int)
    )
    df_merged = df_merged.dropna(subset=["Survival.time", "Event"])

    exclude_cols = list(df_clin.columns) + ["PatientID", id_col, "Event"]
    feature_cols = [c for c in df_merged.columns if c not in exclude_cols]
    feature_cols = (
        df_merged[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )

    return (
        df_merged[feature_cols],
        df_merged["Survival.time"].values,
        df_merged["Event"].values,
    )
