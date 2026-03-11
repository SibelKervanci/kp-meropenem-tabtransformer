# Expected Output Files

Running `pipeline.py` generates the following output files in the working directory.

## Figures (PNG, 300–600 DPI)

| File | Description |
|------|-------------|
| `FIG3_ROC_combined.png` | ROC curves — internal (A) and external (B) evaluation |
| `FIG_CM_combined.png` | Confusion matrices — internal (A) and external (B) |
| `FIG6_attention_pairs.png` | Top pairwise genomic interactions from attention weights |
| `FIG5_shap.png` | SHAP feature importance in transformer embedding space |
| `FIG2_TSNE.png` | t-SNE visualization of transformer-derived embeddings |
| `embeded.png` | Pairwise Pearson correlation heatmap across 32 latent dimensions |

## Console Output

The script prints the following tables to stdout:

- Base classifier benchmarking results (Table 2)
- Hybrid architecture comparison (Table 3)
- External validation metrics (Table 4)
- Per-BioProject performance breakdown
- Bootstrap 95% confidence intervals (internal and external)
- Raw data baseline CatBoost results (ablation study)
- DeLong AUROC comparison p-value

## Supplementary Data

- **Supplementary Table S1**: Full mapping of 32 latent embedding dimensions to genomic features (generated during SHAP + correlation analysis)
- **Supplementary Table S2**: Bootstrap statistics for all evaluated metrics
