# Contributing to kp-meropenem-tabtransformer

Thank you for your interest in contributing to this project. This repository accompanies the manuscript:

> Kervancı, I.S. (2026). *A Leakage-Aware Genomic Prediction Pipeline for Meropenem Resistance in Klebsiella pneumoniae Using Transformer-Based Resistome Representation Learning.* Bioinformatics Advances, Oxford University Press.

---

## How to Contribute

### Reporting Bugs

If you encounter an error while running the pipeline:

1. Open an [Issue](https://github.com/SibelKervanci/kp-meropenem-tabtransformer/issues)
2. Include the following information:
   - Operating system and Python version
   - Full error message and traceback
   - Minimal example that reproduces the issue

### Suggesting Improvements

We welcome suggestions for:
- Extension to other ESKAPE pathogens or antimicrobial classes
- Alternative embedding architectures
- Additional evaluation metrics or visualisations

Please open an Issue with the label `enhancement` before submitting a pull request.

### Submitting a Pull Request

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and ensure the pipeline runs end-to-end
4. Verify that `pipeline.py` passes a syntax check: `python -m py_compile pipeline.py`
5. Submit a pull request with a clear description of your changes

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Add docstrings to all new functions (NumPy docstring format preferred)
- Add inline comments for non-obvious steps
- Use `random_state=42` for all stochastic components to ensure reproducibility

---

## Reproducibility Requirements

Any contribution that modifies model training must:
- Preserve the clade-aware train/test split (`GroupShuffleSplit` with `random_state=42`)
- Preserve safe deduplication logic (conflicting genome signatures must be excluded)
- Report MCC, AUROC, F1, and Perm_MCC on the internal test set
- Not use external validation data during hyperparameter selection

---

## Contact

For questions related to the manuscript or dataset, please contact:

**Ilkay Sibel Kervancı**  
Gaziantep University, Department of Computer Engineering  
sibelkervanci@gantep.edu.tr

---

## Citation

If you use this code or dataset in your work, please cite:

```
Kervancı, I.S. (2026). A Leakage-Aware Genomic Prediction Pipeline for Meropenem
Resistance in Klebsiella pneumoniae Using Transformer-Based Resistome Representation
Learning. Bioinformatics Advances, Oxford University Press.
https://github.com/SibelKervanci/kp-meropenem-tabtransformer
```
