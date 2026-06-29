# Data Layout

The project expects competition and external molecular datasets to be stored locally under `data/`.

```text
data/
|-- train.csv
|-- test.csv
|-- sample_submission.csv
|-- ChEMBL_ASK1(IC50).csv
|-- Pubchem_ASK1.csv
`-- CAS_KPBMA_MAP3K5_IC50s.xlsx
```

These files are intentionally excluded from Git. Keep generated submissions, checkpoints, model weights, cached feature matrices, TensorBoard logs, and AutoGluon/CatBoost outputs outside version control.
