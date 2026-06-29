# Source Audit

This document records how the public repository was checked against the original local workspace.

## Original Workspace

Original folder:

```text
original-drug-prediction-workspace/
|-- fine_tuning.ipynb
|-- roberta_0818.ipynb
|-- Untitled.ipynb
|-- xgboost_0722.py
|-- submission.py
|-- xgboost_0722.zip
|-- submission.zip
|-- submission.7z
|-- open/
|   |-- train.csv
|   |-- test.csv
|   `-- sample_submission.csv
|-- logs/
`-- results/
```

## Public Mapping

| Original item | Public item | Notes |
| --- | --- | --- |
| `xgboost_0722.py` | `src/xgboost_baseline.py` | Path references were changed from `./open/` to `data/`; generated CSV output is written to `outputs/`. |
| `submission.py` | `src/prediction_pipeline.py` | Path references were changed from `./open/` to `data/`; generated submission files are written to `outputs/`. |
| `fine_tuning.ipynb` | `notebooks/legacy_transformer_finetuning.ipynb` | Notebook outputs were cleared for a clean public repository. |
| `roberta_0818.ipynb` | Excluded | Source file is 0 bytes and contains no notebook JSON. |
| `Untitled.ipynb` | Excluded | Source notebook has no cells. |
| `xgboost_0722.zip` | Excluded | Archive only contains `xgboost_0722.py`, which is already represented by `src/xgboost_baseline.py`. |
| `submission.zip` | Excluded | Archive only contains `submission.py`, which is already represented by `src/prediction_pipeline.py`. |
| `submission.7z` | Excluded | Treated as a generated/compressed artifact, not a source file for the public repository. |
| `open/*.csv` | Excluded | Private competition data. |
| `xgboost_*.csv` | Excluded | Generated prediction/submission outputs. |
| `logs/` | Excluded | TensorBoard/local training logs. |
| `results/` | Excluded | Local training/checkpoint output directory. |

## Verification Notes

- The original `results/` directory did not contain additional public result files during the audit.
- The public repository keeps result figures under `assets/`.
- The `.gitignore` excludes data, models, checkpoints, logs, compressed archives, and generated CSV files.
- No raw data or trained model weights are required to view the README and result figures.
