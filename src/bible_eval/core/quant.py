from __future__ import annotations

import re
from typing import Optional

# Approximate bits-per-weight for common GGUF quantization labels, used to order
# a quant-fidelity sweep from highest precision (left) to lowest (right). Values
# are nominal, for ranking only — not exact storage costs.
_QUANT_BPW = {
    "f32": 32.0,
    "fp32": 32.0,
    "f16": 16.0,
    "fp16": 16.0,
    "bf16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.6,
    "q5_k_m": 5.7,
    "q5_k_s": 5.5,
    "q5_0": 5.5,
    "q5_1": 5.9,
    "q4_k_m": 4.8,
    "q4_k_s": 4.6,
    "q4_0": 4.5,
    "q4_1": 4.9,
    "q3_k_l": 3.9,
    "q3_k_m": 3.7,
    "q3_k_s": 3.4,
    "q2_k": 2.6,
}

_QUANT_RE = re.compile(r"(?:^|[-_:.])((?:f|fp|bf)(?:16|32)|q\d(?:_[a-z0-9]+)*)", re.IGNORECASE)


def normalize_quant(label: Optional[str]) -> Optional[str]:
    """Lower-case and trim a quant label (e.g. 'Q4_K_M' -> 'q4_k_m')."""
    if not label:
        return None
    return label.strip().lower() or None


def parse_quant_from_tag(tag: Optional[str]) -> Optional[str]:
    """Best-effort extraction of a quant label from an Ollama/GGUF model tag.

    'gemma3:4b-it-q4_K_M' -> 'q4_k_m'; 'qwen3:0.6b-fp16' -> 'fp16'.
    Returns None when no recognizable quant token is present.
    """
    if not tag:
        return None
    matches = _QUANT_RE.findall(tag)
    return normalize_quant(matches[-1]) if matches else None


def quant_bpw(label: Optional[str]) -> Optional[float]:
    """Nominal bits-per-weight for a quant label, for ordering. None if unknown.

    Falls back to the leading digit after 'q' for unrecognized q-variants
    (e.g. an exotic 'q4_k_xl' -> ~4.0) so new labels still sort sensibly.
    """
    q = normalize_quant(label)
    if not q:
        return None
    if q in _QUANT_BPW:
        return _QUANT_BPW[q]
    m = re.match(r"q(\d)", q)
    if m:
        return float(m.group(1))
    return None
