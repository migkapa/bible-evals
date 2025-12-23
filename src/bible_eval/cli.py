from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

from bible_eval.core.scorer import ErrorCategory, ScoreConfig, Scorer
from bible_eval.core.statistics import (
    compute_confidence_interval,
    compute_metrics_summary,
    compute_proportion_ci,
    format_ci,
)
from bible_eval.data.loader import Taxonomy, VerseDatabase
from bible_eval.engine.interrogator import Interrogator
from bible_eval.engine.sampler import Sampler, SampleConfig
from bible_eval.results.store import append_run, load_history, save_history, write_run_copy


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_score(args: argparse.Namespace) -> int:
    scorer = Scorer(ScoreConfig())
    result = scorer.score_pair(gt=args.gt, pred=args.pred)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    # Prefer the canonical "<think>...</think>\n\nFINAL" pattern.
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    # Fallback: remove any complete blocks; keep remainder.
    out = re.sub(_THINK_BLOCK_RE, "", text)
    out = out.replace("<think>", "").replace("</think>", "")
    return out.strip()


def _postprocess_prediction(pred_raw: str, model_cfg: dict) -> tuple[str, dict]:
    connector = model_cfg.get("connector")
    opts = model_cfg.get("options") or {}
    applied = {}

    pred = pred_raw
    if connector == "ollama" and bool(opts.get("strip_thinking", False)):
        pred2 = _strip_thinking(pred)
        applied["strip_thinking"] = True
        applied["strip_thinking_changed"] = (pred2 != pred)
        pred = pred2

    return pred, applied


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)

    taxonomy = Taxonomy.from_path(cfg["data"]["taxonomy_path"])
    version_key = cfg["eval"]["version"]
    version_cfg = cfg["data"]["versions"][version_key]
    db = VerseDatabase.from_raw_json(
        raw_path=version_cfg["raw_path"],
        taxonomy=taxonomy,
        version=version_key,
    )

    sample_cfg = SampleConfig(
        count=int(cfg["eval"]["sample"]["count"]),
        seed=int(cfg["eval"]["sample"].get("seed", 1)),
        stratified=bool(cfg["eval"]["sample"].get("stratified", False)),
    )
    verses = Sampler(sample_cfg).sample(db)

    prompt_mode = cfg["eval"].get("prompt", "system2")
    scorer = Scorer(ScoreConfig())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_cfg in cfg.get("models", []):
        interrogator = Interrogator.from_model_config(cfg, model_cfg=model_cfg, prompt_mode=prompt_mode)

        results = []
        for verse in verses:
            req, pred, latency = interrogator.query_with_latency(
                verse=verse, version_name=version_cfg.get("name", version_key)
            )
            pred_raw = pred
            pred_clean, applied = _postprocess_prediction(pred_raw, model_cfg=model_cfg)
            scored = scorer.score_pair(gt=verse.text, pred=pred_clean)
            results.append(
                {
                    "verse": verse.to_dict(),
                    "prediction": pred_clean,
                    "prediction_raw": pred_raw,
                    "prompt": {"system": req.system, "user": req.user},
                    "postprocess": applied,
                    "scores": asdict(scored),
                    "prompt_mode": prompt_mode,
                    "model": interrogator.model_name,
                    "version": version_key,
                    "latency_seconds": latency,
                }
            )

        slug = "".join(ch if ch.isalnum() else "_" for ch in interrogator.model_name).strip("_") or "model"
        model_dir = out_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / "results.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        # Legacy labels for backwards compatibility
        labels = [r["scores"].get("label", "unknown") for r in results]
        verbatim = sum(1 for x in labels if x == "verbatim")
        verbatim_with_extras = sum(1 for x in labels if x == "verbatim_with_extras")
        hallucinations = sum(1 for x in labels if x == "total_hallucination")
        inaccurate = len(labels) - verbatim - verbatim_with_extras - hallucinations
        chattery = sum(1 for r in results if float(r["scores"].get("chatter_ratio", 0.0)) > 0.15)
        truncated = sum(1 for r in results if float(r["scores"].get("chatter_ratio", 0.0)) < -0.15)

        # New detailed error categories
        error_cats = [r["scores"].get("error_category", "") for r in results]
        n_minor_deviation = sum(1 for c in error_cats if c == ErrorCategory.MINOR_DEVIATION)
        n_omission = sum(1 for c in error_cats if c == ErrorCategory.OMISSION)
        n_paraphrase = sum(1 for c in error_cats if c == ErrorCategory.PARAPHRASE)
        n_partial_recall = sum(1 for c in error_cats if c == ErrorCategory.PARTIAL_RECALL)

        # Latency statistics
        latencies = [r.get("latency_seconds", 0.0) for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        latency_ci = compute_confidence_interval(latencies)

        # Semantic similarity statistics
        semantic_sims = [r["scores"].get("semantic_similarity", 0.0) for r in results]
        avg_semantic_sim = sum(semantic_sims) / len(semantic_sims) if semantic_sims else 0.0
        semantic_sim_ci = compute_confidence_interval(semantic_sims)

        strip_thinking_enabled = bool((model_cfg.get("options") or {}).get("strip_thinking", False))
        strip_thinking_changed = sum(
            1 for r in results if bool((r.get("postprocess") or {}).get("strip_thinking_changed", False))
        )
        raw_has_think = sum(1 for r in results if "<think>" in str(r.get("prediction_raw", "")).lower())

        strict_hits = sum(1 for r in results if r["scores"]["wer"] == 0.0 and r["scores"]["cer"] == 0.0)
        wer_values = [r["scores"]["wer"] for r in results]
        cer_values = [r["scores"]["cer"] for r in results]
        tsr_values = [r["scores"]["token_sort_ratio"] for r in results]
        chatter_values = [r["scores"]["chatter_ratio"] for r in results]

        avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
        avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
        avg_tsr = sum(tsr_values) / len(tsr_values) if tsr_values else 0.0
        avg_chatter = sum(chatter_values) / len(chatter_values) if chatter_values else 0.0
        avg_len_ratio = (1.0 + avg_chatter) if results else 0.0

        # Compute confidence intervals for key metrics
        wer_ci = compute_confidence_interval(wer_values)
        cer_ci = compute_confidence_interval(cer_values)
        strict_acc_ci = compute_proportion_ci(strict_hits, len(results))
        halluc_ci = compute_proportion_ci(hallucinations, len(results))

        strict_acc = (strict_hits / len(results)) if results else 0.0
        halluc_rate = (hallucinations / len(results)) if results else 0.0
        content_acc = ((verbatim + verbatim_with_extras) / len(results)) if results else 0.0
        clean_output_rate = (1.0 - (strip_thinking_changed / len(results))) if results else 0.0

        if strict_acc >= 0.9 and halluc_rate <= 0.05:
            grade = "A"
            headline = "Excellent verbatim recall."
        elif strict_acc >= 0.75 and halluc_rate <= 0.1:
            grade = "B"
            headline = "Strong recall with minor drift."
        elif strict_acc >= 0.5:
            grade = "C"
            headline = "Mixed: some exact, some paraphrase."
        elif strict_acc >= 0.25:
            grade = "D"
            headline = "Mostly paraphrase/partial recall."
        else:
            grade = "F"
            headline = "Low verbatim fidelity."

        notes = []
        if halluc_rate >= 0.3:
            notes.append("Frequently off-target (hallucinations).")
        elif halluc_rate >= 0.1:
            notes.append("Some off-target responses.")
        if avg_tsr >= 90.0 and strict_acc < 0.5:
            notes.append("Often paraphrases (meaning close, wording off).")
        if verbatim_with_extras >= max(1, int(0.25 * len(results))):
            notes.append("Often adds quotes/citations around correct text.")
        if truncated >= max(1, int(0.25 * len(results))):
            notes.append("Often truncates output.")
        if chattery >= max(1, int(0.25 * len(results))):
            notes.append("Often adds extra chatter.")
        if strip_thinking_enabled and strip_thinking_changed:
            notes.append("Contains visible thinking; hidden for scoring.")
        if not notes:
            notes.append("Behaves consistently on this sample.")

        details_rel = f"details/{run_id}/{slug}.json"
        details_path = Path("results") / details_rel
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        summaries.append(
            {
                "model": interrogator.model_name,
                "model_slug": slug,
                "version": version_key,
                "prompt_mode": prompt_mode,
                "n": len(results),
                "strict_accuracy": strict_acc,
                "avg_wer": avg_wer,
                "avg_cer": avg_cer,
                "avg_token_sort_ratio": avg_tsr,
                "avg_chatter_ratio": avg_chatter,
                "avg_len_ratio": avg_len_ratio,
                "verbatim_count": verbatim,
                "verbatim_with_extras_count": verbatim_with_extras,
                "inaccurate_count": inaccurate,
                "hallucination_count": hallucinations,
                "chattery_count": chattery,
                "truncated_count": truncated,
                "content_accuracy": content_acc,
                "clean_output_rate": clean_output_rate,
                "strip_thinking_enabled": strip_thinking_enabled,
                "strip_thinking_changed_count": strip_thinking_changed,
                "raw_has_think_count": raw_has_think,
                "grade": grade,
                "headline": headline,
                "notes": notes,
                "results_path": str(out_path),
                "details_path": str(details_path),
                "details_rel": details_rel,
                # New: detailed error categories
                "error_categories": {
                    "minor_deviation": n_minor_deviation,
                    "omission": n_omission,
                    "paraphrase": n_paraphrase,
                    "partial_recall": n_partial_recall,
                },
                # New: confidence intervals
                "confidence_intervals": {
                    "strict_accuracy": strict_acc_ci.to_dict() if strict_acc_ci else None,
                    "hallucination_rate": halluc_ci.to_dict() if halluc_ci else None,
                    "avg_wer": wer_ci.to_dict() if wer_ci else None,
                    "avg_cer": cer_ci.to_dict() if cer_ci else None,
                    "avg_latency": latency_ci.to_dict() if latency_ci else None,
                    "avg_semantic_similarity": semantic_sim_ci.to_dict() if semantic_sim_ci else None,
                },
                # New: latency and semantic similarity
                "avg_latency_seconds": avg_latency,
                "avg_semantic_similarity": avg_semantic_sim,
            }
        )

        print(f"Wrote {out_path}")
        # Enhanced output with confidence intervals
        acc_ci_str = format_ci(strict_acc_ci, as_percent=True) if strict_acc_ci else f"{strict_acc*100:.1f}%"
        print(
            f"{interrogator.model_name}: exact={strict_hits}/{len(results)} ({acc_ci_str}) "
            f"halluc={hallucinations}/{len(results)} avg_wer={avg_wer:.3f} "
            f"latency={avg_latency:.2f}s"
        )

    run_summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": version_key,
        "prompt_mode": prompt_mode,
        "sample": asdict(sample_cfg),
        "models": summaries,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")

    # Durable archive (commit-friendly).
    history_path = Path("results") / "history.json"
    append_run(history_path, run_summary)
    write_run_copy(Path("results") / "runs", run_id, run_summary)

    # Keep static site data in sync (if you publish `docs/`).
    site_history_path = Path("docs") / "data" / "history.json"
    save_history(site_history_path, load_history(history_path))

    # Copy latest run details into the site for example rendering.
    for m in summaries:
        rel = m.get("details_rel")
        if not rel:
            continue
        src = Path("results") / rel
        dst = Path("docs") / "data" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


def cmd_export_site(args: argparse.Namespace) -> int:
    history_path = Path(args.history)
    out_dir = Path(args.out)
    runs = load_history(history_path)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "history.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'data' / 'history.json'}")

    for run in runs:
        for m in run.get("models", []) or []:
            rel = m.get("details_rel")
            if not rel:
                continue
            src = Path("results") / rel
            if not src.exists():
                continue
            dst = out_dir / "data" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    run_id = args.run_id
    model_slug = args.model_slug
    details_path = Path("results") / "details" / run_id / f"{model_slug}.json"
    if not details_path.exists():
        raise SystemExit(f"Missing details file: {details_path}")

    entries = json.loads(details_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"Invalid details file: {details_path}")

    def is_bad(e: dict) -> bool:
        label = (e.get("scores") or {}).get("label")
        return label != "verbatim"

    bad = [e for e in entries if is_bad(e)]
    bad.sort(key=lambda e: float((e.get("scores") or {}).get("wer", 0.0)), reverse=True)
    n = min(int(args.count), len(bad))

    for e in bad[:n]:
        verse = e.get("verse") or {}
        ref = verse.get("ref") or f"{verse.get('book', '?')} {verse.get('chapter', '?')}:{verse.get('verse', '?')}"
        scores = e.get("scores") or {}
        print(f"\n{ref}  label={scores.get('label')}  wer={scores.get('wer'):.3f}  cer={scores.get('cer'):.3f}")
        prompt = e.get("prompt") or {}
        if prompt:
            sys_p = prompt.get("system")
            usr_p = prompt.get("user")
            if sys_p:
                print("SYSTEM:", str(sys_p).replace("\n", "\\n"))
            if usr_p:
                print("USER:  ", str(usr_p).replace("\n", "\\n"))
        print(f"GT:   {verse.get('text','')}")
        if "prediction_raw" in e:
            print(f"PRED(raw): {e.get('prediction_raw','')}")
        print(f"PRED(scored): {e.get('prediction','')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bible-eval", description="Evaluate LLM verbatim Bible recall.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_score = sub.add_parser("score", help="Score a single ground-truth/prediction pair.")
    s_score.add_argument("--gt", required=True, help="Ground truth verse text.")
    s_score.add_argument("--pred", required=True, help="Model output text.")
    s_score.set_defaults(func=cmd_score)

    s_run = sub.add_parser("run", help="Run an evaluation from a config.yaml.")
    s_run.add_argument("--config", required=True, help="Path to config.yaml.")
    s_run.set_defaults(func=cmd_run)

    s_site = sub.add_parser("export-site", help="Export results history to a static site data folder.")
    s_site.add_argument("--history", default="results/history.json", help="Path to results history JSON.")
    s_site.add_argument("--out", default="docs", help="Site output directory (writes <out>/data/history.json).")
    s_site.set_defaults(func=cmd_export_site)

    s_ex = sub.add_parser("examples", help="Print bad examples for a given run/model.")
    s_ex.add_argument("--run-id", required=True, help="Run ID, e.g. 20251214T030555Z.")
    s_ex.add_argument(
        "--model-slug",
        required=True,
        help="Model slug from summary, e.g. ollama_llama3_2 or reference_verbatim.",
    )
    s_ex.add_argument("--count", type=int, default=5, help="How many bad examples to print.")
    s_ex.set_defaults(func=cmd_examples)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
