# Blood-Based Biomarkers for Tissue Pathology (GTEx)

Can **Whole Blood gene expression** serve as a non-invasive proxy to detect **organ-specific tissue pathology**? This project uses GTEx v10 matched multi-tissue and blood expression data to build and evaluate blood-based classifiers for pathology conditions across 27 tissues.

## Key Findings

- **54 tissue x pathology models** evaluated via leak-free 5-fold grouped cross-validation
- **10 models achieved AUC >= 0.65**, including:
  - Breast gynecomastoid (AUC = 0.873)
  - Liver cirrhosis (AUC = 0.795)
  - Spleen congestion (AUC = 0.779)
- NLP-based label imputation recovered ~30% additional liver pathology labels
- Random Forest with variance-filtered features (20K genes) compared against Logistic Regression baseline

## Installation

```bash
# Clone
git clone https://github.com/Cranjit9/gtex_gene_expression.git
cd gtex_gene_expression

# Create conda environment
conda env create -f env.yaml
conda activate gtex_biomarkers

# Or use pip
pip install -r requirements.txt
```

## Data

Download the following files from [GTEx Portal](https://gtexportal.org/) and place them in `data/raw/`:

| File | Description |
|------|-------------|
| `GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct` | Gene TPM expression matrix (all tissues) |
| `GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt` | Sample-level metadata |
| `Gtex_restricted.txt` | Donor-level restricted data (AGE, SEX) — requires dbGaP approved access; contact GTEx to request project-specific authorization |
| `meta_data_with_url.csv` | Tissue pathology categories and notes |

## Usage

Run the notebooks in order:

```
notebooks/
├── 01_data_loading_exploration.ipynb   # Load GTEx data, filter blood, PCA
├── 02_liver_steatosis_binary.ipynb     # Binary steatosis model (proof of concept)
├── 03_nlp_imputation.ipynb             # Impute missing pathology labels from notes
├── 04_liver_multicategory.ipynb        # All liver pathology categories
├── 05_all_tissue_models.ipynb          # Pan-tissue LR models (baseline)
├── 06_summary_baseline.ipynb           # Summary figures (bar chart, heatmap)
├── 07_rf_variance_filter.ipynb         # Random Forest + 20K variance filter
├── 08_comparison.ipynb                 # LR vs RF comparison
└── 09_confounder_analysis.ipynb        # Confounder-only vs expression+confounder RF
```

All notebooks import from the `gtex_biomarkers/` package — shared data loading, model training, and evaluation code.

## Project Structure

```
gtex_gene_expression/
├── README.md
├── LICENSE                        # Apache 2.0
├── env.yaml                       # Conda environment
├── requirements.txt               # Pip dependencies
├── gtex_biomarkers/               # Python package
│   ├── __init__.py
│   ├── config.py                  # Centralised parameters & paths
│   ├── data.py                    # Data loading, blood matrix, variance filter
│   ├── labels.py                  # Donor labels, NLP imputation, pair discovery
│   ├── models.py                  # CV pipelines (LR, RF), feature selection
│   ├── evaluation.py              # ROC, PR, CM, boxplot, summary plots
│   └── utils.py                   # Parallel runners, comparison tables
├── notebooks/                     # Analysis notebooks (run in order)
├── data/                          # Not tracked in git
│   ├── raw/                       # GTEx downloads
│   └── processed/                 # Imputed labels
└── output/                        # Not tracked in git
    ├── figures/                   # PDF plots
    └── tables/                    # CSV results
```

## Methodology

- **Cross-validation**: 5-fold `StratifiedGroupKFold` grouped by donor SUBJID (prevents leakage)
- **Feature selection**: Per-fold AUC-based ranking, top 100 genes (train-only)
- **Models**: Logistic Regression (baseline), Random Forest (500 trees, balanced classes)
- **Threshold tuning**: Youden's J statistic (maximises sensitivity + specificity)
- **NLP imputation**: Regex-based category extraction with ConText-inspired negation detection

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2025 Sanju Sinha, Sanford Burnham Prebys Medical Discovery Institute.
