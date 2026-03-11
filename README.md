# Leakage-Aware Genomic Prediction Pipeline for Meropenem Resistance in *Klebsiella pneumoniae*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5547--1860-green.svg)](https://orcid.org/0000-0001-5547-1860)

**A Leakage-Aware Genomic Prediction Pipeline for Meropenem Resistance in *Klebsiella pneumoniae* Using Transformer-Based Resistome Representation Learning**

> Ilkay Sibel Kervancı — Gaziantep University, Department of Computer Engineering  
> Submitted to *Bioinformatics* (Oxford University Press), 2026

---

## Overview

This repository contains the full source code, preprocessing pipeline, and evaluation framework for the **TabTransformer–CatBoost** hybrid architecture described in the manuscript. The framework predicts meropenem resistance in *Klebsiella pneumoniae* from binary gene presence–absence profiles (resistome data), using a self-attention encoder to generate latent genomic embeddings followed by gradient-boosted classification.

### Key features

- **Leakage-aware partitioning**: Clade-aware train/test splitting via Hamming-distance-based agglomerative clustering to prevent phylogenetic information leakage
- **Conflict filtering & deduplication**: Removes genomic profiles with discordant phenotype labels before training
- **Transformer-based resistome embedding**: TabTransformer encoder (d=32, 4 heads, 3 layers) learns contextual gene co-occurrence patterns
- **Multi-objective hyperparameter optimization**: NSGA-II, TPE, and CMA-ES via Optuna, jointly optimizing F1 and MCC
- **External multi-cohort validation**: Evaluated on 7 independent BioProjects (n=305 isolates)
- **Interpretability**: SHAP attribution on latent embeddings + t-SNE visualization + attention weight analysis

---

## Repository Structure

```
├── pipeline.py                  # Main pipeline script (full analysis)
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── data/
│   ├── README_data.md           # Data access instructions (BV-BRC)
│   └── feature_list.txt         # List of 195 AMR genomic markers used
├── results/
│   └── README_results.md        # Description of expected outputs
└── README.md
```

> **Note on data files:** Raw genomic data (`asilverisetigenler.csv`, `validationPRJlison.csv`) are not included in this repository due to size. See `data/README_data.md` for download instructions from BV-BRC.

---

## Installation

### Option 1 — pip

```bash
git clone https://github.com/[KULLANICI-ADI]/[REPO-ADI].git
cd [REPO-ADI]
pip install -r requirements.txt
```

### Option 2 — Conda (recommended)

```bash
git clone https://github.com/[KULLANICI-ADI]/[REPO-ADI].git
cd [REPO-ADI]
conda env create -f environment.yml
conda activate amr-tabtransformer
```

---

## Data Preparation

All genomic data were retrieved from the **Bacterial and Viral Bioinformatics Resource Center (BV-BRC)**:

- **Discovery cohort**: NCBI BioProject [PRJNA376414](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA376414) — Houston Methodist Hospital clinical *K. pneumoniae* collection (n=1,268 after QC)
- **External validation**: 7 independent BioProjects (n=305 isolates) — see manuscript Table 1 for accession numbers

Feature extraction: Binary AMR gene presence–absence profiles were generated using the BV-BRC AMR annotation pipeline. The final feature space comprises **195 curated AMR determinants** including blaKPC variants, porin mutations, and efflux pump components. See `data/feature_list.txt` for the complete list.

Input file format (`;`-separated CSV):

```
Genome ID;gene_1;gene_2;...;gene_195;Resistant Phenotype
GCA_001234;1;0;...;1;Resistant
GCA_005678;0;1;...;0;Susceptible
```

---

## Usage

Run the full pipeline end-to-end:

```bash
python pipeline.py
```

The script executes the following steps in order:

1. Data loading and label encoding
2. Conflict filtering and genomic deduplication
3. Clade-aware train/test split (Hamming distance, threshold=0.05)
4. Base classifier benchmarking (LR, RF, MLP, XGBoost, CatBoost)
5. TabTransformer encoder training (120 epochs, Adam, lr=1e-3)
6. Hybrid TabTransformer–CatBoost evaluation
7. Multi-objective hyperparameter optimization (NSGA-II / TPE / CMA-ES, 50 trials each)
8. External validation on independent cohort
9. Bootstrap confidence intervals (1,000 iterations)
10. Ablation study (raw data baseline)
11. DeLong AUROC comparison test
12. SHAP attribution analysis
13. t-SNE latent space visualization
14. Attention weight extraction and pairwise interaction analysis
15. Embedding dimension correlation heatmap

All figures are saved as high-resolution PNG files (300–600 DPI).

---

## Reproducibility

All random states are fixed:

```python
torch.manual_seed(42)
np.random.seed(42)
# Optuna samplers: seed=42
```

GPU is used automatically if available (`torch.cuda.is_available()`); CPU fallback is supported.

---

## Results Summary

| Model | Accuracy | F1 | MCC | AUROC |
|---|---|---|---|---|
| Logistic Regression | 0.8951 | 0.7606 | 0.7093 | 0.9032 |
| Baseline CatBoost | 0.8642 | 0.6667 | 0.6160 | 0.9135 |
| TabTransformer–CatBoost | 0.9259 | 0.8500 | 0.8013 | 0.8702 |
| **Chained Hybrid (NSGA-II)** | **0.9259** | **0.8537** | **0.8041** | **0.8670** |

External validation (n=199 unique profiles): AUROC=0.8105, F1=0.7552, MCC=0.4624

---

## Citation

If you use this code or pipeline in your research, please cite:

```bibtex
@article{kervanci2026leakage,
  title     = {A Leakage-Aware Genomic Prediction Pipeline for Meropenem Resistance
               in \textit{Klebsiella pneumoniae} Using Transformer-Based
               Resistome Representation Learning},
  author    = {Kervancı, Ilkay Sibel},
  journal   = {Bioinformatics},
  year      = {2026},
  publisher = {Oxford University Press},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Ilkay Sibel Kervancı**  
Department of Computer Engineering, Gaziantep University  
📧 skervanci@gantep.edu.tr  
🔗 ORCID: [0000-0001-5547-1860](https://orcid.org/0000-0001-5547-1860)
