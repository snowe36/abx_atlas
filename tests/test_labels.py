import pandas as pd

from abxatlas.config import ACTIVE_PCHEMBL
from abxatlas.data.annotate import annotate_frame, bucket_text


def test_active_cutoff_constant():
    assert ACTIVE_PCHEMBL == 5.0


def test_assay_description_upgrades_envelope():
    df = pd.DataFrame(
        {
            "target_name": ["Unchecked", "DNA gyrase"],
            "assay_description": [
                "Inhibition of E. coli LpxC",
                "Inhibition of DNA gyrase",
            ],
            "organism": ["Escherichia coli", "Escherichia coli"],
        }
    )
    out = annotate_frame(df)
    assert out.loc[0, "moa_bucket"] == "cell_envelope"
    assert out.loc[1, "moa_bucket"] == "other"


def test_bucket_text_envelope_phrase():
    assert bucket_text("cell wall biosynthesis inhibitor") == "cell_envelope"
