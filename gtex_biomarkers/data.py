"""Data loading, filtering, and blood expression matrix construction."""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from gtex_biomarkers.config import Config


def load_raw_data(cfg=None):
    """Load the four input files and return them as DataFrames.

    Returns
    -------
    df_expr : DataFrame  — gene TPM expression (genes x samples)
    df_samples : DataFrame — sample-level metadata
    df_age : DataFrame — donor-level restricted data (AGE, SEX, …)
    df_meta_url : DataFrame — pathology metadata with tissue info
    """
    cfg = cfg or Config

    df_expr = pd.read_csv(cfg.EXPR_FILE, sep="\t", skiprows=2)
    df_samples = pd.read_csv(cfg.META_FILE, sep="\t")
    df_age = pd.read_csv(cfg.AGE_FILE, sep="\t")
    df_meta_url = pd.read_csv(cfg.PATHOLOGY_FILE)

    return df_expr, df_samples, df_age, df_meta_url


def filter_whole_blood(df_samples):
    """Filter sample metadata to Whole Blood only.

    Returns
    -------
    blood_meta : DataFrame — rows where SMTSD == 'Whole Blood'
    """
    blood_meta = df_samples[df_samples["SMTSD"] == "Whole Blood"].copy()
    return blood_meta


def build_blood_expression_matrix(df_expr, blood_meta):
    """Build (samples x genes) expression matrix for Whole Blood.

    Returns
    -------
    X_wb : DataFrame — shape (n_blood_samples, n_genes), index = SAMPID
    df_blood_wb : DataFrame — subset of expression table with Name/Description
    """
    expr_sample_cols = list(df_expr.columns[2:])
    whole_blood_ids = set(blood_meta["SAMPID"].astype(str))
    overlap_wb = [sid for sid in expr_sample_cols if sid in whole_blood_ids]

    df_blood_wb = df_expr[["Name", "Description"] + overlap_wb].copy()

    expr_numeric = df_blood_wb[overlap_wb].copy()
    expr_numeric.index = df_blood_wb["Name"].astype(str)

    X_wb = expr_numeric.T.astype(float)

    return X_wb, df_blood_wb


def variance_filter(X_wb, n_top=None):
    """Keep only the top-N highest-variance genes.

    Parameters
    ----------
    X_wb : DataFrame — (samples x genes)
    n_top : int — number of genes to keep (default: Config.N_TOP_VAR_GENES)

    Returns
    -------
    X_wb_var : DataFrame — (samples x n_top)
    gene_var : Series — variance per gene, sorted descending
    """
    n_top = n_top or Config.N_TOP_VAR_GENES
    gene_var = X_wb.var(axis=0).sort_values(ascending=False)
    top_genes = gene_var.head(n_top).index.tolist()
    X_wb_var = X_wb[top_genes]
    return X_wb_var, gene_var


CONFOUNDER_COLS = ["SEX", "AGE", "RACE", "DTHHRDY", "TRISCHD"]


def build_confounder_matrix(df_age, blood_subjid):
    """Build donor-level confounder matrix aligned to blood samples.

    Features: SEX, AGE, RACE, DTHHRDY (Hardy Scale), TRISCHD (ischemic time).
    RACE codes 98/99 are treated as missing.  All NaNs imputed with column median.

    Parameters
    ----------
    df_age : DataFrame — donor-level restricted data (one row per donor)
    blood_subjid : Series — index = SAMPID, values = donor SUBJID

    Returns
    -------
    X_conf : DataFrame — shape (n_blood_samples, n_confounders), index = SAMPID
    """
    conf = df_age.drop_duplicates("SUBJID").set_index("SUBJID")
    cols = [c for c in CONFOUNDER_COLS if c in conf.columns]
    conf = conf[cols].copy()

    # Clean RACE: 98/99 = unknown → NaN
    if "RACE" in conf.columns:
        conf.loc[conf["RACE"].isin([98, 99]), "RACE"] = np.nan

    # Map to blood samples via SUBJID
    X_conf = pd.DataFrame(index=blood_subjid.index)
    for col in cols:
        X_conf[col] = blood_subjid.map(conf[col])

    # Impute missing with column median
    X_conf = X_conf.astype(float)
    X_conf = X_conf.fillna(X_conf.median())

    return X_conf


def build_blood_subjid(X_wb):
    """Map each Whole Blood sample ID to donor SUBJID.

    SAMPID format: GTEX-XXXX-..., SUBJID = first two parts joined by '-'.

    Returns
    -------
    blood_subjid : Series — index = SAMPID, values = SUBJID
    """
    blood_subjid = (
        pd.Series(X_wb.index, index=X_wb.index)
        .astype(str)
        .str.split("-").str[:2].str.join("-")
    )
    return blood_subjid


# ── Cache helpers ────────────────────────────────────────────────────────────

_CACHE_FILE = "processed_data.pkl"


def save_cache(X_wb, blood_subjid, blood_meta, df_meta_url, df_age, cfg=None):
    """Save processed data objects to a single pickle file in CACHE_DIR.

    Call this at the end of notebook 01 after building all core objects.
    """
    cfg = cfg or Config
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.CACHE_DIR / _CACHE_FILE
    payload = {
        "X_wb": X_wb,
        "blood_subjid": blood_subjid,
        "blood_meta": blood_meta,
        "df_meta_url": df_meta_url,
        "df_age": df_age,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Cache saved → {cache_path}  ({size_mb:.1f} MB)")


def load_cache(cfg=None):
    """Load processed data from cache.

    Returns
    -------
    X_wb : DataFrame — (samples × genes)
    blood_subjid : Series — SAMPID → SUBJID mapping
    blood_meta : DataFrame — Whole Blood sample metadata
    df_meta_url : DataFrame — pathology metadata
    df_age : DataFrame — donor-level restricted data

    Raises
    ------
    FileNotFoundError
        If cache does not exist — run notebook 01 first.
    """
    cfg = cfg or Config
    cache_path = cfg.CACHE_DIR / _CACHE_FILE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_path}.\n"
            "Please run notebook 01_data_loading_exploration first to build the cache."
        )
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    print(f"Loaded cache from {cache_path}")
    print(f"  X_wb: {payload['X_wb'].shape[0]} samples × {payload['X_wb'].shape[1]} genes")
    return (
        payload["X_wb"],
        payload["blood_subjid"],
        payload["blood_meta"],
        payload["df_meta_url"],
        payload["df_age"],
    )
