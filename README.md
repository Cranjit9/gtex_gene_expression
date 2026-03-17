# GTEx Gene Expression

This repository contains a GTEx-based exploratory analysis workflow for gene expression and tissue pathology work.

## Project Layout

```text
gtex_gene_expression/
├── data/
│   ├── processed/
│   └── raw/
├── output/
│   ├── figures/
│   └── tables/
├── scripts/
│   └── exploratory.ipynb
├── requirements.txt
└── .gitignore
```

## Tracked Files

The repository currently tracks only the project code and configuration:

- `README.md`
- `.gitignore`
- `requirements.txt`
- `scripts/exploratory.ipynb`

Large data files, derived outputs, and local environment files are ignored through `.gitignore`.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place input files under:

- `data/raw/`

Optional derived datasets can be stored under:

- `data/processed/`

Generated figures and tables can be stored under:

- `output/figures/`
- `output/tables/`

## Notes

- The main analysis notebook is `scripts/exploratory.ipynb`.
- The `data/` directory is ignored from version control.
- The `output/` directory is ignored from version control.
