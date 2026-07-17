from abxatlas.data.annotate import bucket_target, gram_stain


def test_envelope_mur():
    assert bucket_target("UDP-N-acetylmuramate--L-alanine ligase MurC") == "cell_envelope"


def test_envelope_lpxc():
    assert bucket_target("UDP-3-O-acyl-N-acetylglucosamine deacetylase LpxC") == "cell_envelope"


def test_exclude_betalactamase():
    assert bucket_target("Beta-lactamase") == "other"


def test_gram_organisms():
    assert gram_stain("Escherichia coli") == "gram_negative"
    assert gram_stain("Staphylococcus aureus") == "gram_positive"
    assert gram_stain("Homo sapiens") == "unknown"
