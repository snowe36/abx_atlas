"""Atlas figure color palette (see atlas-figure-palette skill)."""

from __future__ import annotations

PALETTE = {
    "sage": "#8FB996",
    "teal": "#5FA8A8",
    "peach": "#F4A261",
    "coral": "#E76F51",
    "lavender": "#B8A9E8",
    "mustard": "#E9C46A",
    "navy": "#264653",
}

TEXT = PALETTE["navy"]
MUTED = "#5A6F74"
GRID = "#D9E2E0"
FACE = "#FFFFFF"
SPINE = "#A8B8B5"

MODEL_COLORS = {
    "logreg": PALETTE["navy"],
    "rf": PALETTE["peach"],
    "gbdt": PALETTE["teal"],
    "gnn": PALETTE["sage"],
    "chemberta": PALETTE["lavender"],
}

SPLIT_COLORS = {
    "random": PALETTE["sage"],
    "scaffold": PALETTE["teal"],
    "time": PALETTE["coral"],
}

CAT_COLORS = {
    "cell_envelope": PALETTE["teal"],
    "other": PALETTE["peach"],
    "unknown": MUTED,
    "True": PALETTE["sage"],
    "False": PALETTE["coral"],
    "1": PALETTE["sage"],
    "0": PALETTE["coral"],
    "Active": PALETTE["sage"],
    "Inactive": PALETTE["coral"],
    "No label": GRID,
    "Gram− only": PALETTE["teal"],
    "Gram+ only": PALETTE["peach"],
    "Both": PALETTE["sage"],
    "Neither / unknown": GRID,
}

POS = PALETTE["sage"]
NEG = PALETTE["coral"]
MID = PALETTE["mustard"]
SERIES = [
    PALETTE["teal"],
    PALETTE["sage"],
    PALETTE["mustard"],
    PALETTE["peach"],
    PALETTE["coral"],
    PALETTE["lavender"],
]
