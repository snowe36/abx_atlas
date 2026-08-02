# abx_atlas demo (5 minutes)

See the leakage story **without** downloading ChEMBL.

## Run

```bash
# from repo root (venv recommended)
bash demo/run_demo.sh
# or:
abx-demo
```

## What you get

| Output | Meaning |
|--------|---------|
| `outputs/fixture_leakage_rocauc.png` | Miniature random / scaffold / time gap on fixture assays |
| `outputs/fig3_leakage_rocauc.png` | Full ChEMBL hero (when committed metrics are present) |
| `outputs/fixture_leakage_results.csv` | Per-model ROC-AUC on the fixture splits |

## Requirements

- Python ≥ 3.11 + RDKit
- `pip install -e ".[dev]"` (the script does this)
- CPU only

## Not in the demo

Full `scripts/reproduce.sh` (ChEMBL download → atlas → QSAR). GPU GNN / ChemBERTa are optional and separate.
