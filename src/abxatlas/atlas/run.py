"""Run chemical-space atlas end-to-end."""

from __future__ import annotations

import json
import logging

from abxatlas.atlas.plots import (
    plot_fig1_chemspace_atlas,
    plot_fig2_scaffold_diversity,
    plot_gram_label_balance,
    plot_np_envelope_chemspace,
    plot_np_vs_synthetic_moa,
    plot_pca_chemspace,
    plot_scaffold_counts,
)
from abxatlas.atlas.stats import summarize_atlas
from abxatlas.data.curate import load_curated
from abxatlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def run_atlas() -> dict:
    ensure_dirs()
    _, compounds = load_curated()
    summary = summarize_atlas(compounds)
    summary_path = PROCESSED / "atlas_summary.csv"
    summary.to_csv(summary_path, index=False)

    figures = []
    # Primary portfolio figures
    for fn in (plot_fig1_chemspace_atlas, plot_fig2_scaffold_diversity):
        try:
            figures.append(str(fn(compounds)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", fn.__name__, exc)

    for color_by in ("moa_bucket", "is_natural_product", "gram_neg_active"):
        try:
            figures.append(str(plot_pca_chemspace(compounds, color_by=color_by)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("PCA plot (%s) failed: %s", color_by, exc)
    for fn in (
        plot_np_envelope_chemspace,
        plot_scaffold_counts,
        plot_np_vs_synthetic_moa,
        plot_gram_label_balance,
    ):
        try:
            figures.append(str(fn(compounds)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", fn.__name__, exc)

    meta = {
        "n_compounds": int(len(compounds)),
        "summary_csv": str(summary_path),
        "figures": figures,
        "figures_dir": str(FIGURES),
    }
    meta_path = PROCESSED / "atlas_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Atlas complete: %s", meta)
    print(summary.to_string(index=False))
    return meta
