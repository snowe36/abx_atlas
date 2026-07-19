"""Named chemotype families, temporal failure enrichment, and surprise case studies."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from scipy.stats import fisher_exact

from abxatlas.config import MORGAN_NBITS, MORGAN_RADIUS
from abxatlas.models.interpret import error_labels, predict_binary
from abxatlas.paths import FIGURES, PROCESSED, RESOURCES, ensure_dirs

logger = logging.getLogger(__name__)

AMIDE_SMARTS = Chem.MolFromSmarts("C(=O)N")


@lru_cache(maxsize=1)
def load_chemotype_families(path: Path | None = None) -> list[dict]:
    yaml_path = path or (RESOURCES / "chemotype_families.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get("families") or [])


def _compile_smarts(smarts_list: list[str] | None) -> list:
    pats = []
    for sma in smarts_list or []:
        pat = Chem.MolFromSmarts(sma)
        if pat is None:
            logger.warning("Invalid SMARTS skipped: %s", sma)
            continue
        pats.append(pat)
    return pats


def _morgan_fp(mol) -> object | None:
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)


def _family_match(mol, family: dict, ref_fps: dict[str, object]) -> bool:
    """Return True if mol belongs to this family (SMARTS / heuristics / similarity)."""
    if mol is None:
        return False

    smarts_pats = family.get("_smarts_pats") or []
    require_pats = family.get("_require_pats") or []
    min_mw = family.get("min_mw")
    min_amide = family.get("min_amide_count")
    sim_thr = family.get("similarity_threshold")
    fid = family["id"]

    mw = Descriptors.MolWt(mol) if (min_mw is not None or min_amide is not None) else None
    n_amide = (
        len(mol.GetSubstructMatches(AMIDE_SMARTS))
        if (min_amide is not None or require_pats)
        else None
    )

    # Structural heuristic path (glycopeptide / polymyxin-style)
    if require_pats or min_mw is not None or min_amide is not None:
        ok = True
        if min_mw is not None and (mw is None or mw < float(min_mw)):
            ok = False
        if min_amide is not None and (n_amide is None or n_amide < int(min_amide)):
            ok = False
        if require_pats and not all(mol.HasSubstructMatch(p) for p in require_pats):
            ok = False
        if smarts_pats and not any(mol.HasSubstructMatch(p) for p in smarts_pats):
            # When SMARTS are listed alongside heuristics, require at least one SMARTS hit
            ok = False
        if ok and (require_pats or min_mw is not None):
            return True

    # Pure SMARTS families
    if smarts_pats and not (require_pats or min_mw is not None or min_amide is not None):
        if any(mol.HasSubstructMatch(p) for p in smarts_pats):
            return True

    # Similarity fallback to a reference drug
    if sim_thr is not None and fid in ref_fps:
        fp = _morgan_fp(mol)
        if fp is not None:
            tan = DataStructs.TanimotoSimilarity(fp, ref_fps[fid])
            if tan >= float(sim_thr):
                return True

    return False


def _prepare_families(families: list[dict] | None = None) -> tuple[list[dict], dict[str, object]]:
    fams = []
    ref_fps: dict[str, object] = {}
    for raw in families or load_chemotype_families():
        fam = dict(raw)
        fam["_smarts_pats"] = _compile_smarts(fam.get("smarts"))
        fam["_require_pats"] = _compile_smarts(fam.get("require_smarts"))
        ref_smi = fam.get("similarity_reference_smiles") or fam.get("reference_smiles")
        if ref_smi:
            ref_mol = Chem.MolFromSmiles(ref_smi)
            fp = _morgan_fp(ref_mol)
            if fp is not None:
                ref_fps[fam["id"]] = fp
        fams.append(fam)
    return fams, ref_fps


def assign_chemotype_families(
    smiles: list[str] | pd.Series,
    families: list[dict] | None = None,
) -> pd.DataFrame:
    """Assign primary + multi-label chemotype families to each SMILES.

    Returns one row per input molecule with columns:
      smiles, primary_family, primary_label, families (pipe-joined ids)
    """
    fams, ref_fps = _prepare_families(families)
    rows = []
    for smi in list(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        hits = [f for f in fams if _family_match(mol, f, ref_fps)]
        primary = hits[0] if hits else None
        rows.append(
            {
                "smiles": smi,
                "primary_family": primary["id"] if primary else "other",
                "primary_label": primary["label"] if primary else "Other / unclassified",
                "families": "|".join(f["id"] for f in hits) if hits else "",
            }
        )
    return pd.DataFrame(rows)


def chemotype_error_enrichment(
    smiles: list[str] | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    min_count: int = 5,
    families: list[dict] | None = None,
) -> pd.DataFrame:
    """Per-family FP/FN rates on a held-out set, with Fisher enrichment vs rest."""
    assigned = assign_chemotype_families(smiles, families=families)
    err = error_labels(y_true, y_pred)
    frame = assigned.copy()
    frame["y_true"] = np.asarray(y_true).astype(int)
    frame["y_pred"] = np.asarray(y_pred).astype(int)
    frame["error"] = err
    frame["is_error"] = frame["error"].isin(["FP", "FN"])
    frame["is_fn"] = frame["error"] == "FN"
    frame["is_fp"] = frame["error"] == "FP"

    overall_err = float(frame["is_error"].mean()) if len(frame) else 0.0
    rows = []
    for fam_id, g in frame.groupby("primary_family"):
        n = len(g)
        if n < min_count:
            continue
        fp = int((g["error"] == "FP").sum())
        fn = int((g["error"] == "FN").sum())
        tp = int((g["error"] == "TP").sum())
        tn = int((g["error"] == "TN").sum())
        err_n = fp + fn
        err_rate = err_n / n
        # Fisher exact: family×error vs rest
        rest = frame[frame["primary_family"] != fam_id]
        table = [
            [err_n, n - err_n],
            [int(rest["is_error"].sum()), int((~rest["is_error"]).sum())],
        ]
        try:
            odds, pval = fisher_exact(table, alternative="greater")
        except ValueError:
            odds, pval = float("nan"), float("nan")
        label = g["primary_label"].iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "family": fam_id,
                "label": label,
                "n": n,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "fp_rate": fp / n,
                "fn_rate": fn / n,
                "error_rate": err_rate,
                "overall_error_rate": overall_err,
                "error_lift": (err_rate / overall_err) if overall_err > 0 else float("nan"),
                "fisher_odds_ratio": float(odds),
                "fisher_p_greater": float(pval),
                "active_rate": float(g["y_true"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["error_lift", "n"], ascending=[False, False])
        .reset_index(drop=True)
    )


def plot_chemotype_enrichment(
    enrichment: pd.DataFrame,
    split_name: str,
    out_path: Path | None = None,
    top_n: int = 10,
) -> Path | None:
    """Horizontal bars: error lift by named chemotype family."""
    ensure_dirs()
    if enrichment is None or enrichment.empty:
        return None
    sub = enrichment[enrichment["split_name"] == split_name].copy()
    if sub.empty:
        return None
    # Prefer named families over "other"
    sub = sub[sub["family"] != "other"]
    if sub.empty:
        return None
    top = sub.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#922b21" if r > 1.15 else "#1b4f72" for r in top["error_lift"]]
    ax.barh(top["label"], top["error_lift"], color=colors)
    ax.axvline(1.0, color="#666666", lw=0.9, ls="--")
    ax.set_xlabel("Error-rate lift vs overall held-out set")
    title_split = "temporal" if split_name == "time" else split_name
    ax.set_title(f"Figure 7. Chemotypes enriched in {title_split}-split failures")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["error_lift"] + 0.03,
            i,
            f"n={int(row['n'])}  err={row['error_rate']:.0%}  p={row['fisher_p_greater']:.2g}",
            va="center",
            fontsize=8,
            color="#333333",
        )
    ax.set_xlim(0, max(2.0, float(top["error_lift"].max()) * 1.25))
    fig.tight_layout()
    default = (
        FIGURES / "fig7_chemotype_temporal_failures.png"
        if split_name == "time"
        else FIGURES / f"fig7_chemotype_{split_name}_failures.png"
    )
    out = out_path or default
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def _nearest_train_tanimoto(
    X_train: np.ndarray,
    X_query: np.ndarray,
) -> np.ndarray:
    """Max Tanimoto of each query row to the training fingerprint matrix."""
    from sklearn.metrics import pairwise_distances

    if len(X_train) == 0 or len(X_query) == 0:
        return np.full(len(X_query), np.nan)
    dist = pairwise_distances(X_query.astype(bool), X_train.astype(bool), metric="jaccard")
    return 1.0 - dist.min(axis=1)


def surprise_case_study(
    smiles: list[str] | pd.Series,
    y: np.ndarray,
    X: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    model,
    case_family_ids: tuple[str, ...] = ("fluoroquinolone", "glycopeptide", "polymyxin"),
    families: list[dict] | None = None,
) -> pd.DataFrame:
    """Per historic family: train support, split membership, errors, OOD distance."""
    fams, _ = _prepare_families(families)
    label_map = {f["id"]: f.get("label", f["id"]) for f in fams}
    assigned = assign_chemotype_families(smiles, families=families)
    # Multi-label membership for case-study families (not only primary)
    membership = {
        fid: assigned["families"].fillna("").str.contains(rf"(?:^|\\|){fid}(?:\\||$)", regex=True)
        | (assigned["primary_family"] == fid)
        for fid in case_family_ids
    }

    rows = []
    for fid in case_family_ids:
        mask = membership[fid].to_numpy()
        idxs = np.flatnonzero(mask)
        n_total = int(len(idxs))
        for split_name, (tr, te) in splits.items():
            tr_set, te_set = set(map(int, tr)), set(map(int, te))
            in_train = np.array([i in tr_set for i in idxs], dtype=bool)
            in_test = np.array([i in te_set for i in idxs], dtype=bool)
            train_idx = idxs[in_train]
            test_idx = idxs[in_test]

            y_pred_test = np.array([], dtype=int)
            err_test = np.array([], dtype=object)
            nn_tani = np.array([], dtype=float)
            if len(test_idx) > 0:
                _, y_pred_test = predict_binary(model, X[test_idx])
                err_test = error_labels(y[test_idx], y_pred_test)
                nn_tani = _nearest_train_tanimoto(X[tr], X[test_idx])

            n_te = int(len(test_idx))
            fp = int((err_test == "FP").sum()) if n_te else 0
            fn = int((err_test == "FN").sum()) if n_te else 0
            tp = int((err_test == "TP").sum()) if n_te else 0
            tn = int((err_test == "TN").sum()) if n_te else 0
            rows.append(
                {
                    "family": fid,
                    "label": label_map.get(fid, fid),
                    "split_name": split_name,
                    "n_total": n_total,
                    "n_train": int(len(train_idx)),
                    "n_test": n_te,
                    "train_fraction": (len(train_idx) / n_total) if n_total else float("nan"),
                    "test_active_rate": float(y[test_idx].mean()) if n_te else float("nan"),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "accuracy": ((tp + tn) / n_te) if n_te else float("nan"),
                    "error_rate": ((fp + fn) / n_te) if n_te else float("nan"),
                    "mean_nn_tanimoto": float(np.nanmean(nn_tani)) if n_te else float("nan"),
                    "ood_hint": (
                        "scarce_in_task"
                        if n_total < 10
                        else (
                            "held_out_chemotype"
                            if n_te > 0 and len(train_idx) == 0
                            else (
                                "structurally_remote"
                                if n_te > 0 and float(np.nanmean(nn_tani)) < 0.35
                                else "supported"
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_surprise_case_study(
    case_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path | None:
    """Figure 8: three historic families × split — support, errors, OOD distance."""
    ensure_dirs()
    if case_df is None or case_df.empty:
        return None

    families = list(dict.fromkeys(case_df["label"].tolist()))
    split_order = [s for s in ("scaffold", "time", "random") if s in set(case_df["split_name"])]
    if not families or not split_order:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=False)
    panel_colors = {"scaffold": "#1b4f72", "time": "#922b21", "random": "#117a65"}

    for ax, fam_label in zip(axes, families, strict=False):
        sub = case_df[case_df["label"] == fam_label]
        xs = np.arange(len(split_order))
        n_test = [
            int(sub.loc[sub["split_name"] == s, "n_test"].iloc[0])
            if s in set(sub["split_name"])
            else 0
            for s in split_order
        ]
        err = [
            float(sub.loc[sub["split_name"] == s, "error_rate"].iloc[0])
            if s in set(sub["split_name"]) and pd.notna(
                sub.loc[sub["split_name"] == s, "error_rate"].iloc[0]
            )
            else 0.0
            for s in split_order
        ]
        nn = [
            float(sub.loc[sub["split_name"] == s, "mean_nn_tanimoto"].iloc[0])
            if s in set(sub["split_name"]) and pd.notna(
                sub.loc[sub["split_name"] == s, "mean_nn_tanimoto"].iloc[0]
            )
            else 0.0
            for s in split_order
        ]
        n_train = [
            int(sub.loc[sub["split_name"] == s, "n_train"].iloc[0])
            if s in set(sub["split_name"])
            else 0
            for s in split_order
        ]
        n_total = int(sub["n_total"].iloc[0]) if len(sub) else 0

        width = 0.35
        ax.bar(
            xs - width / 2,
            err,
            width=width,
            color=[panel_colors.get(s, "#555") for s in split_order],
            label="Error rate",
        )
        ax.bar(
            xs + width / 2,
            nn,
            width=width,
            color="#b9770e",
            alpha=0.85,
            label="Mean NN Tanimoto",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(split_order)
        ax.set_ylim(0, 1.05)
        ax.set_title(fam_label, fontsize=11)
        hint = sub["ood_hint"].iloc[0] if len(sub) else ""
        ax.text(
            0.02,
            0.98,
            f"n={n_total} in task\ntrain/test (scaffold): {n_train[0] if n_train else 0}/{n_test[0] if n_test else 0}\n{hint}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#333333",
        )
        for i, (e, t, nt) in enumerate(zip(err, nn, n_test, strict=True)):
            if nt == 0:
                ax.text(i, 0.5, "no test\nmembers", ha="center", va="center", fontsize=8, color="#666")
            else:
                ax.text(i - width / 2, e + 0.03, f"{e:.0%}", ha="center", fontsize=7)
                ax.text(i + width / 2, t + 0.03, f"{t:.2f}", ha="center", fontsize=7)

    axes[0].set_ylabel("Rate / Tanimoto")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(
        "Figure 8. Surprise case study — historic scaffolds the model treats as OOD",
        y=1.12,
        fontsize=12,
    )
    fig.tight_layout()
    out = out_path or (FIGURES / "fig8_surprise_case_study.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def run_chemotype_interpretation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    smiles: list[str] | pd.Series,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Enrichment on scaffold + time splits, plus three-family surprise case study."""
    ensure_dirs()
    enrich_frames = []
    figures: list[str] = []

    for split_name in ("scaffold", "time"):
        if split_name not in splits:
            continue
        tr, te = splits[split_name]
        if len(np.unique(y[te])) < 1:
            continue
        # Fit-free: caller passes a model already fit on the relevant train split
        # For multi-split enrichment we re-fit logreg per split inside the runner.
        _, y_pred = predict_binary(model, X[te])
        enrich = chemotype_error_enrichment(
            pd.Series(list(smiles)).iloc[te],
            y[te],
            y_pred,
            split_name=split_name,
        )
        if not enrich.empty:
            enrich_frames.append(enrich)
            fig = plot_chemotype_enrichment(enrich, split_name=split_name)
            if fig is not None:
                figures.append(str(fig))

    enrich_df = pd.concat(enrich_frames, ignore_index=True) if enrich_frames else pd.DataFrame()
    enrich_path = PROCESSED / "qsar_chemotype_enrichment.csv"
    if not enrich_df.empty:
        enrich_df.to_csv(enrich_path, index=False)
    else:
        enrich_path = None

    case_df = surprise_case_study(smiles, y, X, splits, model)
    case_path = PROCESSED / "qsar_surprise_case_study.csv"
    case_df.to_csv(case_path, index=False)
    fig_case = plot_surprise_case_study(case_df)
    if fig_case is not None:
        figures.append(str(fig_case))

    # Narrative bullets for meta / README hooks
    narrative = _build_narrative(enrich_df, case_df)

    return {
        "chemotype_enrichment_csv": str(enrich_path) if enrich_path else None,
        "surprise_case_study_csv": str(case_path),
        "figures": figures,
        "narrative": narrative,
        "n_families_enriched": int(enrich_df["family"].nunique()) if not enrich_df.empty else 0,
    }


def _build_narrative(enrich_df: pd.DataFrame, case_df: pd.DataFrame) -> dict:
    """Short discovery-style bullets from enrichment + case study tables."""
    out: dict = {"temporal_failures": [], "case_study": []}
    if enrich_df is not None and not enrich_df.empty:
        time = enrich_df[
            (enrich_df["split_name"] == "time") & (enrich_df["family"] != "other")
        ].copy()
        time = time.sort_values("error_lift", ascending=False)
        for _, row in time.head(5).iterrows():
            if row["error_lift"] >= 1.1:
                out["temporal_failures"].append(
                    {
                        "family": row["label"],
                        "error_lift": float(row["error_lift"]),
                        "error_rate": float(row["error_rate"]),
                        "n": int(row["n"]),
                        "p": float(row["fisher_p_greater"]),
                    }
                )
    if case_df is not None and not case_df.empty:
        for fam, g in case_df.groupby("label"):
            scaf = g[g["split_name"] == "scaffold"]
            time = g[g["split_name"] == "time"]
            out["case_study"].append(
                {
                    "family": fam,
                    "n_total": int(g["n_total"].iloc[0]),
                    "scaffold_error_rate": (
                        float(scaf["error_rate"].iloc[0]) if len(scaf) else None
                    ),
                    "time_error_rate": float(time["error_rate"].iloc[0]) if len(time) else None,
                    "scaffold_nn_tanimoto": (
                        float(scaf["mean_nn_tanimoto"].iloc[0]) if len(scaf) else None
                    ),
                    "ood_hint": str(scaf["ood_hint"].iloc[0]) if len(scaf) else None,
                }
            )
    return out
