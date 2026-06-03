from bible_eval.core.quant import normalize_quant, parse_quant_from_tag, quant_bpw


def test_normalize_quant():
    assert normalize_quant("Q4_K_M") == "q4_k_m"
    assert normalize_quant("  FP16 ") == "fp16"
    assert normalize_quant("") is None
    assert normalize_quant(None) is None


def test_parse_quant_from_tag():
    assert parse_quant_from_tag("gemma3:4b-it-q4_K_M") == "q4_k_m"
    assert parse_quant_from_tag("gemma3:1b-it-fp16") == "fp16"
    assert parse_quant_from_tag("qwen3:0.6b-q8_0") == "q8_0"
    assert parse_quant_from_tag("llama3.2:latest") is None
    assert parse_quant_from_tag(None) is None


def test_quant_bpw_known_order():
    # Higher precision must rank above lower precision.
    assert quant_bpw("fp16") > quant_bpw("q8_0") > quant_bpw("q4_k_m") > quant_bpw("q2_k")


def test_quant_bpw_unknown_variant_falls_back_to_leading_digit():
    # An exotic q4 variant should still sort near 4 bits.
    assert quant_bpw("q4_k_xl") == 4.0
    assert quant_bpw("totally-unknown") is None
