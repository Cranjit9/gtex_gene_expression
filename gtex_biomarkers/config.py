"""Centralised configuration for the GTEx blood-based biomarkers pipeline."""

from pathlib import Path


class Config:
    """All tuneable parameters in one place."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = ROOT_DIR / "data" / "raw"
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    TABLES_DIR = ROOT_DIR / "output" / "tables"
    FIGURES_DIR = ROOT_DIR / "output" / "figures"

    # ── Input files ───────────────────────────────────────────────────────────
    EXPR_FILE = RAW_DIR / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct"
    META_FILE = RAW_DIR / "GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
    AGE_FILE = RAW_DIR / "Gtex_restricted.txt"
    PATHOLOGY_FILE = RAW_DIR / "meta_data_with_url.csv"

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED = 0
    N_SPLITS = 5

    # ── Feature selection ─────────────────────────────────────────────────────
    TOP_K_FEATURES = 100
    N_TOP_VAR_GENES = 20_000

    # ── Sample thresholds ─────────────────────────────────────────────────────
    SAMPLE_THRESHOLD_LIVER = 5
    ALL_TISSUE_THRESHOLD = 50
    MIN_POS_NEG_BLOOD = 5

    # ── Model parameters ──────────────────────────────────────────────────────
    LR_SOLVER = "saga"
    LR_MAX_ITER = 5_000
    RF_N_ESTIMATORS = 500
    RF_MAX_FEATURES = "sqrt"

    # ── Evaluation ────────────────────────────────────────────────────────────
    AUC_CUTOFF = 0.65

    # ── Labels to exclude (normal / healthy) ──────────────────────────────────
    NORMAL_LABELS = {"clean_specimens", "no_abnormalities"}

    @classmethod
    def ensure_dirs(cls):
        """Create output directories if they don't exist."""
        for d in [cls.RAW_DIR, cls.PROCESSED_DIR, cls.TABLES_DIR, cls.FIGURES_DIR]:
            d.mkdir(parents=True, exist_ok=True)
