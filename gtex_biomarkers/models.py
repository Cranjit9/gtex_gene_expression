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
            solver=cfg.LR_SOLVER, max_iter=cfg.LR_MAX_ITER,
            random_state=cfg.SEED,
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


def _compute_oof_threshold(y, oof):
    """Compute Youden's J optimal threshold from OOF predictions."""
    mask = ~np.isnan(oof)
    fpr_oof, tpr_oof, thresholds_oof = roc_curve(
        y.values[mask], oof[mask]
    )
    j_scores = tpr_oof - fpr_oof
    return thresholds_oof[np.argmax(j_scores)]


def run_cv(X, y, groups, model_factory, cfg=None, top_k=None,
           save_features=False, X_extra=None, feature_selection=True):
    """Run leak-free 5-fold grouped CV.

    Unified CV function supporting three modes:
    - feature_selection=True, X_extra=None  → AUC-based FS on X (expression only)
    - feature_selection=False, X_extra=None → no FS, use all columns (e.g. confounders)
    - feature_selection=True, X_extra=df    → AUC-based FS on X, concat X_extra always

    Parameters
    ----------
    X : DataFrame — (samples × genes) or (samples × confounders)
    y : Series — binary labels (0/1)
    groups : Series — donor SUBJID for grouping
    model_factory : callable — returns a fresh model/pipeline per fold
    cfg : Config class
    top_k : int — number of features to select per fold
    save_features : bool — if True, capture selected genes + RF importances
    X_extra : DataFrame — extra columns always included (e.g. confounders)
    feature_selection : bool — whether to run AUC-based feature selection

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

        # Feature selection (train only) or use all columns
        if feature_selection:
            top_feat = _auc_feature_selection(Xtr, ytr, top_k=top_k)
            Xtr_fit = Xtr[top_feat]
            Xte_fit = Xte[top_feat]
        else:
            top_feat = list(X.columns)
            Xtr_fit = Xtr
            Xte_fit = Xte

        # Concatenate extra columns if provided
        if X_extra is not None:
            Xtr_extra, Xte_extra = X_extra.iloc[tr], X_extra.iloc[te]
            Xtr_fit = pd.concat([Xtr_fit, Xtr_extra], axis=1)
            Xte_fit = pd.concat([Xte_fit, Xte_extra], axis=1)

        # Train + predict
        model = model_factory()
        model.fit(Xtr_fit, ytr)
        proba = model.predict_proba(Xte_fit)[:, 1]
        oof[te] = proba

        fauc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(fauc)

        # Capture feature-level info
        if save_features:
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
    optimal_thresh = _compute_oof_threshold(y, oof)

    result = {
        "y": y, "oof": oof,
        "fold_fprs": fold_fprs, "fold_tprs": fold_tprs, "fold_aucs": fold_aucs,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "optimal_threshold": optimal_thresh,
    }
    if save_features:
        result["feature_info"] = feature_info
    return result


# Backwards-compatible aliases
def run_cv_no_fs(X, y, groups, model_factory, cfg=None):
    """Run CV without feature selection (for low-dimensional inputs like confounders)."""
    return run_cv(X, y, groups, model_factory, cfg=cfg, feature_selection=False)


def run_cv_combined(X_expr, X_conf, y, groups, model_factory, cfg=None, top_k=None):
    """Run CV with AUC FS on expression, always including confounders."""
    return run_cv(X_expr, y, groups, model_factory, cfg=cfg, top_k=top_k,
                  X_extra=X_conf)


def run_tissue_models(tissue, cat_list, df_meta_url, blood_subjid, X_wb,
                      model_factory, cfg=None, save_features=False):
    """Run CV for all categories of a single tissue.

    Designed to be called inside joblib.Parallel — one call per tissue.
    """
    from gtex_biomarkers.labels import assign_donor_labels

    cfg = cfg or Config
    results = []

    for cat, n_samples in cat_list:
        tag = f"{tissue} | {cat}"

        y, donor_lab, n_pos, n_neg = assign_donor_labels(
            df_meta_url, tissue, cat, blood_subjid
        )
        keep = y.notna()
        X_cat = X_wb.loc[keep].copy()
        y_cat = y.loc[keep].astype(int)
        g_cat = blood_subjid.loc[keep].astype(str)

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
    from gtex_biomarkers.labels import assign_donor_labels

    cfg = cfg or Config
    conf_results = []
    comb_results = []

    for cat, n_samples in cat_list:
        tag = f"{tissue} | {cat}"

        y, donor_lab, n_pos, n_neg = assign_donor_labels(
            df_meta_url, tissue, cat, blood_subjid
        )
        keep = y.notna()
        X_expr_cat = X_wb.loc[keep].copy()
        X_conf_cat = X_conf.loc[keep].copy()
        y_cat = y.loc[keep].astype(int)
        g_cat = blood_subjid.loc[keep].astype(str)

        if n_pos < cfg.MIN_POS_NEG_BLOOD or n_neg < cfg.MIN_POS_NEG_BLOOD:
            continue

        # Model A: confounders only (no feature selection)
        res_c = run_cv(X_conf_cat, y_cat, g_cat, model_factory, cfg=cfg,
                       feature_selection=False)
        res_c["tissue"] = tissue
        res_c["category"] = cat
        conf_results.append((tag, res_c))

        # Model B: expression + confounders (AUC FS on expression, confounders always kept)
        res_cb = run_cv(X_expr_cat, y_cat, g_cat, model_factory, cfg=cfg,
                        X_extra=X_conf_cat)
        res_cb["tissue"] = tissue
        res_cb["category"] = cat
        comb_results.append((tag, res_cb))

    return conf_results, comb_results
