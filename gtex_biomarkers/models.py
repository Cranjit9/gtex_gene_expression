"""Cross-validation pipelines — shared by LR and RF models."""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, roc_auc_score

from gtex_biomarkers.config import Config


def _auc_feature_selection(Xtr, ytr, top_k=None):
    """Per-gene AUC-based feature selection on training data only.

    For each gene, compute |AUC - 0.5| against the binary label.
    Return the top_k gene column names.
    """
    top_k = top_k or Config.TOP_K_FEATURES
    strengths = {}
    for col in Xtr.columns:
        s = pd.to_numeric(Xtr[col], errors="coerce").fillna(0)
        if s.nunique() < 2:
            strengths[col] = 0.0
            continue
        a = roc_auc_score(ytr, s)
        strengths[col] = abs(a - 0.5)
    scores = pd.Series(strengths).reindex(Xtr.columns).fillna(0)
    return scores.nlargest(top_k).index.tolist()


def make_lr_pipeline(cfg=None):
    """Create a Logistic Regression pipeline with StandardScaler."""
    cfg = cfg or Config
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver=cfg.LR_SOLVER, max_iter=cfg.LR_MAX_ITER
        )),
    ])


def make_rf_model(cfg=None):
    """Create a Random Forest classifier (no scaler needed)."""
    cfg = cfg or Config
    return RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_features=cfg.RF_MAX_FEATURES,
        class_weight="balanced",
        random_state=cfg.SEED,
        n_jobs=1,  # n_jobs=1 inside parallel workers to avoid oversubscription
    )


def run_cv(X, y, groups, model_factory, cfg=None, top_k=None,
           save_features=False):
    """Run leak-free 5-fold grouped CV with per-fold feature selection.

    Parameters
    ----------
    X : DataFrame — (samples × genes)
    y : Series — binary labels (0/1)
    groups : Series — donor SUBJID for grouping
    model_factory : callable — returns a fresh model/pipeline per fold
    cfg : Config class
    top_k : int — number of features to select per fold
    save_features : bool — if True, capture selected genes + RF importances

    Returns
    -------
    dict with keys:
        y, oof, fold_fprs, fold_tprs, fold_aucs,
        mean_auc, std_auc, optimal_threshold
        feature_info (only if save_features=True): list of per-fold dicts
    """
    cfg = cfg or Config
    top_k = top_k or cfg.TOP_K_FEATURES

    cv = StratifiedGroupKFold(
        n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED
    )

    oof = np.full(len(y), np.nan)
    fold_fprs, fold_tprs, fold_aucs = [], [], []
    feature_info = [] if save_features else None

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # Feature selection (train only)
        top_feat = _auc_feature_selection(Xtr, ytr, top_k=top_k)

        # Train + predict
        model = model_factory()
        model.fit(Xtr[top_feat], ytr)
        proba = model.predict_proba(Xte[top_feat])[:, 1]
        oof[te] = proba

        fauc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(fauc)

        # Capture feature-level info
        if save_features:
            # Works for bare RF and Pipeline with a "model" step
            estimator = (model.named_steps["model"]
                         if hasattr(model, "named_steps") else model)
            importances = getattr(estimator, "feature_importances_", None)
            fold_info = {
                "fold": fold,
                "selected_genes": top_feat,
                "rf_importances": (dict(zip(top_feat, importances))
                                   if importances is not None else {}),
            }
            feature_info.append(fold_info)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    # Youden's J optimal threshold
    mask_oof = ~np.isnan(oof)
    fpr_oof, tpr_oof, thresholds_oof = roc_curve(
        y.values[mask_oof], oof[mask_oof]
    )
    j_scores = tpr_oof - fpr_oof
    best_idx = np.argmax(j_scores)
    optimal_thresh = thresholds_oof[best_idx]

    result = {
        "y": y, "oof": oof,
        "fold_fprs": fold_fprs, "fold_tprs": fold_tprs, "fold_aucs": fold_aucs,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "optimal_threshold": optimal_thresh,
    }
    if save_features:
        result["feature_info"] = feature_info
    return result


def run_cv_no_fs(X, y, groups, model_factory, cfg=None):
    """Run CV without feature selection (for low-dimensional inputs like confounders).

    Same grouped CV as run_cv but uses all columns directly — no AUC-based filtering.
    """
    cfg = cfg or Config

    cv = StratifiedGroupKFold(
        n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED
    )

    oof = np.full(len(y), np.nan)
    fold_fprs, fold_tprs, fold_aucs = [], [], []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        model = model_factory()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        oof[te] = proba

        fauc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(fauc)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    mask_oof = ~np.isnan(oof)
    fpr_oof, tpr_oof, thresholds_oof = roc_curve(
        y.values[mask_oof], oof[mask_oof]
    )
    j_scores = tpr_oof - fpr_oof
    optimal_thresh = thresholds_oof[np.argmax(j_scores)]

    return {
        "y": y, "oof": oof,
        "fold_fprs": fold_fprs, "fold_tprs": fold_tprs, "fold_aucs": fold_aucs,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "optimal_threshold": optimal_thresh,
    }


def run_cv_combined(X_expr, X_conf, y, groups, model_factory, cfg=None, top_k=None):
    """Run CV with AUC-based feature selection on expression, always including confounders.

    Per fold: select top_k expression features (train-only), concatenate with all
    confounder columns, then train and predict.
    """
    cfg = cfg or Config
    top_k = top_k or cfg.TOP_K_FEATURES

    cv = StratifiedGroupKFold(
        n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED
    )

    oof = np.full(len(y), np.nan)
    fold_fprs, fold_tprs, fold_aucs = [], [], []

    for fold, (tr, te) in enumerate(cv.split(X_expr, y, groups=groups), 1):
        Xtr_e, Xte_e = X_expr.iloc[tr], X_expr.iloc[te]
        Xtr_c, Xte_c = X_conf.iloc[tr], X_conf.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # Feature selection on expression only
        top_feat = _auc_feature_selection(Xtr_e, ytr, top_k=top_k)

        # Concatenate selected expression + all confounders
        Xtr_combined = pd.concat([Xtr_e[top_feat], Xtr_c], axis=1)
        Xte_combined = pd.concat([Xte_e[top_feat], Xte_c], axis=1)

        model = model_factory()
        model.fit(Xtr_combined, ytr)
        proba = model.predict_proba(Xte_combined)[:, 1]
        oof[te] = proba

        fauc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(fauc)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    mask_oof = ~np.isnan(oof)
    fpr_oof, tpr_oof, thresholds_oof = roc_curve(
        y.values[mask_oof], oof[mask_oof]
    )
    j_scores = tpr_oof - fpr_oof
    optimal_thresh = thresholds_oof[np.argmax(j_scores)]

    return {
        "y": y, "oof": oof,
        "fold_fprs": fold_fprs, "fold_tprs": fold_tprs, "fold_aucs": fold_aucs,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "optimal_threshold": optimal_thresh,
    }


def run_tissue_models(tissue, cat_list, df_meta_url, blood_subjid, X_wb,
                      model_factory, cfg=None, save_features=False):
    """Run CV for all categories of a single tissue.

    Designed to be called inside joblib.Parallel — one call per tissue.

    Parameters
    ----------
    tissue : str — tissue name
    cat_list : list of (category, n_samples) tuples
    df_meta_url : DataFrame — pathology metadata
    blood_subjid : Series — blood SAMPID → donor SUBJID
    X_wb : DataFrame — blood expression matrix
    model_factory : callable — returns a fresh model per fold
    save_features : bool — if True, capture per-fold gene importances

    Returns
    -------
    list of (tag, result_dict) tuples
    """
    cfg = cfg or Config
    results = []

    # Shared tissue metadata
    tissue_sub = df_meta_url[df_meta_url["Tissue"] == tissue].copy()
    tissue_sub["SUBJID"] = (
        tissue_sub["Tissue.Sample.ID"].astype(str)
        .str.split("-").str[:2].str.join("-")
    )
    known = tissue_sub[tissue_sub["Pathology.Categories"].notna()].copy()

    for cat, n_samples in cat_list:
        tag = f"{tissue} | {cat}"

        has_cat = known["Pathology.Categories"].str.contains(
            cat, case=False
        ).astype(int)
        donor_lab = has_cat.groupby(known["SUBJID"]).max()

        y_cat = blood_subjid.map(donor_lab)
        keep = y_cat.notna()
        X_cat = X_wb.loc[keep].copy()
        y_cat = y_cat.loc[keep].astype(int)
        g_cat = blood_subjid.loc[keep].astype(str)

        n_pos = int(y_cat.sum())
        n_neg = int((y_cat == 0).sum())

        if n_pos < cfg.MIN_POS_NEG_BLOOD or n_neg < cfg.MIN_POS_NEG_BLOOD:
            continue

        res = run_cv(X_cat, y_cat, g_cat, model_factory, cfg=cfg,
                     save_features=save_features)
        res["tissue"] = tissue
        res["category"] = cat
        results.append((tag, res))

    return results


def run_tissue_confounder_models(tissue, cat_list, df_meta_url, blood_subjid,
                                 X_wb, X_conf, model_factory, cfg=None):
    """Run confounder-only AND expression+confounder models for one tissue.

    Returns two lists: (conf_results, combined_results), each a list of
    (tag, result_dict) tuples.
    """
    cfg = cfg or Config
    conf_results = []
    comb_results = []

    tissue_sub = df_meta_url[df_meta_url["Tissue"] == tissue].copy()
    tissue_sub["SUBJID"] = (
        tissue_sub["Tissue.Sample.ID"].astype(str)
        .str.split("-").str[:2].str.join("-")
    )
    known = tissue_sub[tissue_sub["Pathology.Categories"].notna()].copy()

    for cat, n_samples in cat_list:
        tag = f"{tissue} | {cat}"

        has_cat = known["Pathology.Categories"].str.contains(
            cat, case=False
        ).astype(int)
        donor_lab = has_cat.groupby(known["SUBJID"]).max()

        y_cat = blood_subjid.map(donor_lab)
        keep = y_cat.notna()
        X_expr_cat = X_wb.loc[keep].copy()
        X_conf_cat = X_conf.loc[keep].copy()
        y_cat = y_cat.loc[keep].astype(int)
        g_cat = blood_subjid.loc[keep].astype(str)

        n_pos = int(y_cat.sum())
        n_neg = int((y_cat == 0).sum())
        if n_pos < cfg.MIN_POS_NEG_BLOOD or n_neg < cfg.MIN_POS_NEG_BLOOD:
            continue

        # Model A: confounders only (no feature selection)
        res_c = run_cv_no_fs(X_conf_cat, y_cat, g_cat, model_factory, cfg=cfg)
        res_c["tissue"] = tissue
        res_c["category"] = cat
        conf_results.append((tag, res_c))

        # Model B: expression + confounders (AUC FS on expression, confounders always kept)
        res_cb = run_cv_combined(X_expr_cat, X_conf_cat, y_cat, g_cat,
                                 model_factory, cfg=cfg)
        res_cb["tissue"] = tissue
        res_cb["category"] = cat
        comb_results.append((tag, res_cb))

    return conf_results, comb_results
