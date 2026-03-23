"""General utility functions."""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from gtex_biomarkers.config import Config
from gtex_biomarkers.models import run_tissue_models, run_tissue_confounder_models


def run_all_tissue_models_parallel(pairs_df, df_meta_url, blood_subjid, X_wb,
                                   model_factory, cfg=None, n_jobs=-1):
    """Run CV models for all tissue × category pairs, parallelized by tissue.

    Parameters
    ----------
    pairs_df : DataFrame — columns: tissue, category, n_samples
    df_meta_url : DataFrame — pathology metadata
    blood_subjid : Series — blood SAMPID → donor SUBJID
    X_wb : DataFrame — blood expression matrix
    model_factory : callable — returns a fresh model per fold
    n_jobs : int — number of parallel workers (-1 = all cores)

    Returns
    -------
    results_dict : dict — {tag: result_dict}
    summary_df : DataFrame — sorted by mean_auc descending
    """
    cfg = cfg or Config

    # Group by tissue
    tissue_groups = {}
    for _, row in pairs_df.iterrows():
        tissue_groups.setdefault(row["tissue"], []).append(
            (row["category"], row["n_samples"])
        )

    # Run in parallel
    parallel_out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_tissue_models)(
            tissue, cat_list, df_meta_url, blood_subjid, X_wb,
            model_factory, cfg=cfg
        )
        for tissue, cat_list in sorted(tissue_groups.items())
    )

    # Collect
    results_dict = {}
    for tissue_results in parallel_out:
        for tag, res in tissue_results:
            results_dict[tag] = res

    # Summary table
    summary_rows = [
        {"tissue": r["tissue"], "category": r["category"],
         "mean_auc": r["mean_auc"], "std_auc": r["std_auc"],
         "optimal_threshold": r["optimal_threshold"]}
        for r in results_dict.values()
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_auc", ascending=False)

    return results_dict, summary_df


def _make_summary(results_dict):
    """Build a summary DataFrame from a results dict."""
    rows = [
        {"tissue": r["tissue"], "category": r["category"],
         "mean_auc": r["mean_auc"], "std_auc": r["std_auc"],
         "optimal_threshold": r["optimal_threshold"]}
        for r in results_dict.values()
    ]
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False)


def run_all_confounder_models_parallel(pairs_df, df_meta_url, blood_subjid,
                                       X_wb, X_conf, model_factory,
                                       cfg=None, n_jobs=-1):
    """Run confounder-only AND expression+confounder RF models, parallelized by tissue.

    Returns
    -------
    conf_results : dict — {tag: result_dict} for confounder-only models
    conf_summary : DataFrame
    comb_results : dict — {tag: result_dict} for expression+confounder models
    comb_summary : DataFrame
    """
    cfg = cfg or Config

    tissue_groups = {}
    for _, row in pairs_df.iterrows():
        tissue_groups.setdefault(row["tissue"], []).append(
            (row["category"], row["n_samples"])
        )

    parallel_out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_tissue_confounder_models)(
            tissue, cat_list, df_meta_url, blood_subjid, X_wb, X_conf,
            model_factory, cfg=cfg
        )
        for tissue, cat_list in sorted(tissue_groups.items())
    )

    conf_results, comb_results = {}, {}
    for tissue_conf, tissue_comb in parallel_out:
        for tag, res in tissue_conf:
            conf_results[tag] = res
        for tag, res in tissue_comb:
            comb_results[tag] = res

    return (conf_results, _make_summary(conf_results),
            comb_results, _make_summary(comb_results))


def build_comparison_table(lr_summary, rf_summary):
    """Merge LR and RF summaries into a comparison table.

    Returns
    -------
    comp : DataFrame — with columns mean_auc_lr, mean_auc_rf, auc_diff
    """
    comp = lr_summary.merge(
        rf_summary, on=["tissue", "category"], suffixes=("_lr", "_rf"), how="outer"
    )
    comp["auc_diff"] = comp["mean_auc_rf"] - comp["mean_auc_lr"]
    comp = comp.sort_values("auc_diff", ascending=False)
    return comp


def top_models_table(summary_df, results_dict, auc_cutoff=None):
    """Filter to models above AUC cutoff and add sample counts.

    Returns
    -------
    top_df : DataFrame — with n_blood_samples, n_positive, prevalence columns
    """
    auc_cutoff = auc_cutoff or Config.AUC_CUTOFF
    top = summary_df[summary_df["mean_auc"] >= auc_cutoff].copy()
    top = top.sort_values("mean_auc", ascending=False).reset_index(drop=True)

    for idx, row in top.iterrows():
        tag = f"{row['tissue']} | {row['category']}"
        if tag in results_dict:
            res = results_dict[tag]
            top.loc[idx, "n_blood_samples"] = len(res["y"])
            top.loc[idx, "n_positive"] = int(res["y"].sum())
            top.loc[idx, "prevalence"] = res["y"].mean()

    if "n_blood_samples" in top.columns:
        top["n_blood_samples"] = top["n_blood_samples"].astype(int)
        top["n_positive"] = top["n_positive"].astype(int)

    return top
