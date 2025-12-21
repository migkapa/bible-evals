#!/usr/bin/env python3
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "results" / "history.json"
OUT_PATH = ROOT / "docs" / "results.svg"


def is_reference_model(model: dict) -> bool:
    name = str(model.get("model") or "")
    slug = str(model.get("model_slug") or "")
    return name.startswith("reference:") or slug.startswith("reference_")


def fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_svg(models: list[dict]) -> str:
    width = 920
    left = 200
    right = 40
    top = 56
    bottom = 34
    bar_h = 22
    gap = 10

    n = len(models)
    inner_h = n * bar_h + max(0, n - 1) * gap
    height = max(180, top + inner_h + bottom)
    inner_w = width - left - right

    title = "Best strict accuracy by model (all runs)"
    subtitle = "Reference baseline hidden"

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Model strict accuracy chart">'
    )
    parts.append(
        "<style>"
        "text{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;}"
        ".title{font-size:18px;font-weight:600;fill:#0f172a;}"
        ".subtitle{font-size:12px;fill:#475569;}"
        ".label{font-size:12px;fill:#0f172a;}"
        ".value{font-size:12px;fill:#0f172a;}"
        ".grid{stroke:#e2e8f0;stroke-width:1;}"
        "</style>"
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')

    parts.append(f'<text x="{left}" y="26" class="title">{html.escape(title)}</text>')
    parts.append(f'<text x="{left}" y="44" class="subtitle">{html.escape(subtitle)}</text>')

    for i in range(6):
        x = left + (inner_w * i) / 5
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + inner_h + 6}" class="grid" />'
        )
        pct = fmt_pct(i / 5)
        parts.append(f'<text x="{x:.1f}" y="{top - 14}" class="subtitle" text-anchor="middle">{pct}</text>')

    if not models:
        parts.append(
            f'<text x="{left}" y="{top + 20}" class="label">No non-reference models found.</text>'
        )
        parts.append("</svg>")
        return "\n".join(parts)

    max_val = max((float(m.get("strict_accuracy") or 0.0) for m in models), default=0.0)
    max_val = max(0.000001, max_val)

    for idx, m in enumerate(models):
        raw_name = str(m.get("model") or "")
        label = raw_name.replace("ollama:", "")
        val = float(m.get("strict_accuracy") or 0.0)
        val = clamp(val, 0.0, 1.0)
        bar_w = (val / max_val) * inner_w
        y = top + idx * (bar_h + gap)
        parts.append(
            f'<text x="{left - 10}" y="{y + bar_h - 6}" class="label" text-anchor="end">{html.escape(label)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="6" fill="#16a34a" />'
        )
        parts.append(
            f'<text x="{left + bar_w + 8:.1f}" y="{y + bar_h - 6}" class="value">{fmt_pct(val)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    if not HISTORY_PATH.exists():
        raise SystemExit(f"Missing history: {HISTORY_PATH}")
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise SystemExit("History is empty.")

    best: dict[str, dict] = {}
    for run in history:
        for m in run.get("models") or []:
            if is_reference_model(m):
                continue
            val = float(m.get("strict_accuracy") or 0.0)
            prev = best.get(m.get("model") or "")
            if prev is None or val > float(prev.get("strict_accuracy") or 0.0):
                best[m.get("model") or ""] = m
    models = list(best.values())
    models.sort(key=lambda m: float(m.get("strict_accuracy") or 0.0), reverse=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_svg(models), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
