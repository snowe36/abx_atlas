# abx_atlas

**Antibacterial chemical-space atlas & leakage-aware QSAR** — a CPU-only, reproducible analysis of public ChEMBL data.

> *Can simple models predict Gram-negative antibacterial labels from structure alone—and how much of that performance is scaffold / time leakage?*

---

## Motivation

My PhD focused on acyltransferases that modify the bacterial **cell envelope**—biology that sits at the interface of antibiotic mechanism and chemical space.

This repository started from that motivation, but public ChEMBL target annotations for envelope-specific MoAs are sparse at organism-level MIC scale. The **main contribution** is therefore a clear, interview-ready evaluation of antibacterial QSAR generalization:

1. Characterize antibacterial chemical space (scaffolds, Gram labels, MoA buckets where available).
2. Train simple CPU models (Morgan FP → logistic regression / random forest).
3. Show that **random splits overstate performance** relative to scaffold and temporal holdouts.
4. Interpret failures (bit weights, FP/FN scaffolds, nearest neighbors).

**Talk track:** *“I know how to evaluate molecular ML properly—and I can show where optimistic chemogenomics numbers come from.”*

---

## Data source and curation

| Item | Detail |
|------|--------|
| Source | [ChEMBL](https://www.ebi.ac.uk/chembl/) activities via `chembl-webresource-client` |
| Organisms | Priority set: *E. coli*, *P. aeruginosa*, *K. pneumoniae*, *A. baumannii*, *S. aureus*, *S. pneumoniae*, *E. faecium*, *M. tuberculosis* |
| Keep | Records with `pchembl_value`; types MIC / IC50 / Ki / EC50 / IZ / Potency |
| Active | pChEMBL ≥ 5 (≈ ≤ 10 µM) |
| Features | RDKit Morgan FP (radius 2, 2048 bits) |
| MoA bucket | Keyword match on target name + assay description → `cell_envelope` / `other` / `unknown` |
| NP flag | ChEMBL `natural_product` (optional; current laptop pull used `--no-np`) |

**Current curated snapshot (priority pull, max 3000 / organism):**

- **8,793** compounds · **3,820** Bemis–Murcko scaffolds (ratio **0.43**)
- **4,061** with Gram− labels · active rate **0.70**
- Envelope-tagged compounds: **129** after keyword + assay-text expansion (still a minority — organism-level MIC assays rarely name molecular targets)

---

## Benchmark design

**Primary task:** binary Gram-negative activity (`gram_neg_active`).

**Models (intentionally simple):**

- Logistic regression (interpretable coefficients)
- Random forest

**Splits (the point of the repo):**

| Split | What it tests |
|-------|----------------|
| Random (stratified) | Optimistic upper bound / leakage-prone |
| Scaffold (Bemis–Murcko) | Generalization to novel chemotypes |
| Time (earliest document year) | Prospective / publication-era shift |

**Interpretation:** top ± Morgan-bit logreg weights, column-shuffle importance, FP/FN scaffold enrichment, nearest-train-neighbor Tanimoto for TP/FP/FN.

No deep learning in v1 — the story is evaluation hygiene, not architecture hunting.

---

## Results

### Chemical space (Figures 1–2)

Antibacterial compounds occupy **broad** Morgan-FP space with substantial scaffold diversity (~43% unique scaffolds). Most scaffolds are rare — exactly the setting where random splits leak.

### Leakage-aware QSAR (Figure 3 — hero)

| Split | LogReg ROC-AUC | RF ROC-AUC |
|-------|---------------:|-----------:|
| Random | **0.78** | **0.86** |
| Scaffold | **0.70** | **0.83** |
| Time | **0.51** | **0.59** |

Mean optimistic gap (random − scaffold): **~0.06**. Temporal holdout nearly collapses linear performance to chance.

### Learning curve (Figure 4)

On held-out scaffolds, RF improves with more train data then **plateaus** (Δ ≈ +0.06 from 10%→100% train); logreg does **not** improve monotonically. More ChEMBL rows alone do not erase chemotype / era bias.

### What the model is learning (Figure 5 + tables)

- False positives are often **structurally close** to actives in the training set (high Tanimoto to nearest neighbor) yet sit on **novel scaffolds** at test time.
- Logreg bit weights and permutation importance highlight associative fingerprint patterns — not causal substructures.

**Takeaway:** apparent Gram− QSAR strength is partly **chemotype memorization** and **dataset era**. Scaffold- and time-aware evaluation makes that visible.

---

## Limitations

- Organism-level assays dominate; molecular MoA labels (esp. cell-envelope) are incomplete.
- Labels conflate potency with permeability, efflux, stability, and assay conditions.
- NP annotations were skipped in the laptop pull (`--no-np`); NP panels may be empty until re-fetched.
- Keyword MoA bucketing is heuristic; β-lactamases are explicitly excluded from the envelope bucket.
- Morgan bits indicate association, not mechanism.

---

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/reproduce.sh
# or:
abx-download --max-per-organism 5000   # add NP flags by omitting --no-np
abx-atlas
abx-qsar
pytest -q
```

| Artifact | Path |
|----------|------|
| Fig 1 chemspace | `reports/figures/fig1_chemspace_atlas.png` |
| Fig 2 scaffolds | `reports/figures/fig2_scaffold_diversity.png` |
| Fig 3 leakage (hero) | `reports/figures/fig3_leakage_rocauc.png` |
| Fig 4 learning curve | `reports/figures/fig4_learning_curve.png` |
| Fig 5 neighbors | `reports/figures/fig5_error_neighbors.png` |
| Metrics | `data/processed/qsar_leakage_results.csv`, `atlas_summary.csv` |

CPU only. MIT license. Cite ChEMBL when using regenerated tables ([CITATION.cff](CITATION.cff)).

---

## Future directions

- Integrate **COCONUT** / NPAtlas natural-product annotations
- Improve **mechanism-level labeling** (target family / UniProt / GO), not only name keywords
- Prospective **external validation** sets beyond ChEMBL time splits
- Stronger classical baselines (e.g. gradient boosting) once leakage story is fixed
- Bit→substructure highlighting for the strongest FP/FN motifs
- Optional later: graph / pretrained molecular embeddings — **after** evaluation is solid

---

## Layout

```
abx_atlas/
  README.md
  pyproject.toml
  src/abxatlas/
    data/          # ChEMBL fetch + cleaning + MoA/NP annotation
    featurize/     # Morgan FP, Bemis–Murcko scaffolds
    atlas/         # chemspace stats + figures
    models/        # splits, QSAR, learning curves, interpretation
    resources/     # envelope keywords, Gram organism lists
  scripts/reproduce.sh
  reports/figures/
  tests/
  LICENSE
```
