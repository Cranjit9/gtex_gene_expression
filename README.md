# Blood-Based Biomarkers for Tissue Pathology (GTEx)

Can **Whole Blood gene expression** serve as a non-invasive proxy to detect **organ-specific tissue pathology**? This project uses GTEx v10 matched multi-tissue and blood expression data to build and evaluate blood-based classifiers for pathology conditions across 27 tissues.

## Key Findings

- **53 tissue x pathology models** evaluated via leak-free 5-fold grouped cross-validation
- **Random Forest on variance-filtered blood expression (20K genes)** outperformed the Logistic Regression baseline on the multi-tissue screen
- **PC + confounder models** identified **9 tissue x pathology pairs** with `AUC >= 0.60` and `ΔAUC >= 0.05` over confounders alone, including:
  - Liver cirrhosis (`AUC = 0.785`, `Δ = 0.218`)
  - Spleen congestion (`AUC = 0.804`, `Δ = 0.083`)
  - Muscle atrophy (`AUC = 0.698`, `Δ = 0.079`)
- **Gene-level back-projection from PC models** produced full ranked biomarker lists for the qualifying pairs
- **Pathway enrichment analysis** on those PC-derived gene rankings identified shared and tissue-specific biology across **9 qualifying pairs** and **3 pathway libraries**
- NLP-based label imputation recovered additional liver pathology labels from free-text notes

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
├── 09_confounder_analysis.ipynb        # Confounder-only vs expression+confounder RF
├── 10_tissue_pc_regression.ipynb       # PCA on blood expression, PC screening, PC+confounder RF
├── 11_pc_gene_importance.ipynb         # Back-project PC importance to gene-level rankings
└── 12_pathway_enrichment.ipynb         # GSEA on PC-derived gene rankings
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
- **Confounder analysis**: Age, sex, race, death circumstances, and ischemic time compared against expression-only and expression+confounder models
- **PC regression**: `StandardScaler -> PCA -> univariate AUC selection of PCs -> Random Forest`, fit inside each CV fold
- **Gene attribution from PCs**: Selected PC importances are back-projected to genes using normalized PCA loadings to obtain per-gene importance proportions
- **Pathway analysis**: GSEA prerank on full PC-derived gene rankings using KEGG 2026, Reactome 2024, and GO Biological Process 2025
- **Threshold tuning**: Youden's J statistic (maximises sensitivity + specificity)
- **NLP imputation**: Regex-based category extraction with ConText-inspired negation detection

## Main Outputs

- `output/tables/cv_results_all_tissue_rf.csv`: RF performance across all tissue x pathology pairs
- `output/tables/cv_three_way_comparison.csv`: confounder-only vs expression-only vs combined comparison
- `output/tables/pc_auc_results.csv`: PC-only and PC+confounder AUC summary
- `output/tables/pc_gene_importance_full.csv`: full ranked gene importance lists for qualifying PC models
- `output/tables/gsea_pathway_enrichment.csv`: pathway enrichment results across qualifying pairs and libraries
- `output/figures/`: summary ROC, PR, delta-AUC, PC, gene-sharing, and pathway figures

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2025 Sanju Sinha, Sanford Burnham Prebys Medical Discovery Institute.
