from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

from bible_eval import __version__
from bible_eval.core.abstention import classify_void_response
from bible_eval.core.scorer import ScoreConfig, Scorer
from bible_eval.core.stats import bootstrap_ci, wilson_interval
from bible_eval.data.loader import Taxonomy, VerseDatabase
from bible_eval.engine.interrogator import Interrogator
from bible_eval.engine.sampler import Sampler, SampleConfig, void_probes
from bible_eval.results.store import append_run, load_history, save_history, write_run_copy

_REFERENCE_CONNECTORS = {"reference", "baseline"}


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _git_state() -> dict:
    """Best-effort git commit + dirty flag for reproducibility. Never raises."""
    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run("rev-parse", "--short", "HEAD")
    status = _run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def _wilson_ci(k: int, n: int) -> dict:
    lo, hi = wilson_interval(int(k), int(n))
    return {"lo": lo, "hi": hi, "method": "wilson"}


def _bootstrap_ci(values: list, *, scale: float = 1.0) -> dict:
    lo, hi = bootstrap_ci([float(v) for v in values])
    return {"lo": lo * scale, "hi": hi * scale, "method": "bootstrap"}


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
    sampler = Sampler(sample_cfg)
    verses = sampler.sample(db)

    prompt_mode = cfg["eval"].get("prompt", "system2")
    scorer = Scorer(ScoreConfig())

    abstention_cfg = cfg["eval"].get("abstention") or {}
    abstention_enabled = bool(abstention_cfg.get("enabled", False))
    abstention_count = int(abstention_cfg.get("count", 0))
    abstention_seed = int(abstention_cfg.get("seed", sample_cfg.seed))
    void_set = (
        void_probes(db, count=abstention_count, seed=abstention_seed)
        if abstention_enabled and abstention_count > 0
        else []
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_cfg in cfg.get("models", []):
        interrogator = Interrogator.from_model_config(cfg, model_cfg=model_cfg, prompt_mode=prompt_mode)

        results = []
        for verse in verses:
            req, pred = interrogator.query_with_request(
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
                }
            )

        slug = "".join(ch if ch.isalnum() else "_" for ch in interrogator.model_name).strip("_") or "model"
        model_dir = out_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / "results.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        labels = [r["scores"].get("label", "unknown") for r in results]
        verbatim = sum(1 for x in labels if x == "verbatim")
        verbatim_with_extras = sum(1 for x in labels if x == "verbatim_with_extras")
        hallucinations = sum(1 for x in labels if x == "total_hallucination")
        inaccurate = len(labels) - verbatim - verbatim_with_extras - hallucinations
        chattery = sum(1 for r in results if float(r["scores"].get("chatter_ratio", 0.0)) > 0.15)
        truncated = sum(1 for r in results if float(r["scores"].get("chatter_ratio", 0.0)) < -0.15)

        strip_thinking_enabled = bool((model_cfg.get("options") or {}).get("strip_thinking", False))
        strip_thinking_changed = sum(
            1 for r in results if bool((r.get("postprocess") or {}).get("strip_thinking_changed", False))
        )
        raw_has_think = sum(1 for r in results if "<think>" in str(r.get("prediction_raw", "")).lower())

        strict_hits = sum(1 for r in results if r["scores"]["wer"] == 0.0 and r["scores"]["cer"] == 0.0)
        avg_wer = sum(r["scores"]["wer"] for r in results) / len(results) if results else 0.0
        avg_cer = sum(r["scores"]["cer"] for r in results) / len(results) if results else 0.0
        avg_tsr = sum(r["scores"]["token_sort_ratio"] for r in results) / len(results) if results else 0.0
        avg_chatter = sum(r["scores"]["chatter_ratio"] for r in results) / len(results) if results else 0.0
        avg_len_ratio = (1.0 + avg_chatter) if results else 0.0

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

        # Abstention probes: ask for verses that do not exist and reward refusal
        # over fabrication. Skipped for the offline reference connector (it has
        # no opinion about void references).
        abstention = None
        if void_set and model_cfg.get("connector") not in _REFERENCE_CONNECTORS:
            probe_results = []
            for probe in void_set:
                req, p_raw = interrogator.query_with_request(
                    verse=probe, version_name=version_cfg.get("name", version_key)
                )
                p_clean, p_applied = _postprocess_prediction(p_raw, model_cfg=model_cfg)
                verdict = classify_void_response(p_clean)
                probe_results.append(
                    {
                        "ref": probe.ref,
                        "prediction": p_clean,
                        "prediction_raw": p_raw,
                        "prompt": {"system": req.system, "user": req.user},
                        "postprocess": p_applied,
                        "verdict": verdict,
                        "model": interrogator.model_name,
                    }
                )
            n_probes = len(probe_results)
            refused = sum(1 for r in probe_results if r["verdict"] == "refused")
            fabricated = sum(1 for r in probe_results if r["verdict"] == "fabricated")
            empties = sum(1 for r in probe_results if r["verdict"] == "empty")
            abstention_rel = f"details/{run_id}/{slug}.abstention.json"
            (Path("results") / abstention_rel).write_text(
                json.dumps(probe_results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            abstention = {
                "n": n_probes,
                "refused": refused,
                "fabricated": fabricated,
                "empty": empties,
                "abstention_rate": (refused / n_probes) if n_probes else 0.0,
                "fabrication_rate": (fabricated / n_probes) if n_probes else 0.0,
                "ci": _wilson_ci(refused, n_probes) if n_probes else None,
                "details_rel": abstention_rel,
            }
            print(
                f"{interrogator.model_name}: abstention refused={refused}/{n_probes} "
                f"fabricated={fabricated}/{n_probes}"
            )

        # Confidence intervals: Wilson for rates, percentile-bootstrap for
        # continuous metrics. Stored under `ci[<metric>] = {lo, hi, method}`
        # so the site can render every point estimate with its uncertainty.
        n_res = len(results)
        clean_hits = n_res - strip_thinking_changed
        ci = {}
        if n_res:
            ci["strict_accuracy"] = _wilson_ci(strict_hits, n_res)
            ci["content_accuracy"] = _wilson_ci(verbatim + verbatim_with_extras, n_res)
            ci["hallucination_rate"] = _wilson_ci(hallucinations, n_res)
            ci["clean_output_rate"] = _wilson_ci(clean_hits, n_res)
            ci["avg_wer"] = _bootstrap_ci([r["scores"]["wer"] for r in results])
            ci["avg_cer"] = _bootstrap_ci([r["scores"]["cer"] for r in results])
            ci["avg_token_sort_ratio"] = _bootstrap_ci(
                [r["scores"]["token_sort_ratio"] for r in results]
            )
            ci["avg_chatter_ratio"] = _bootstrap_ci([r["scores"]["chatter_ratio"] for r in results])

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
                "hallucination_rate": halluc_rate,
                "abstention": abstention,
                "ci": ci,
                "decode_options": dict((model_cfg.get("options") or {})),
                "connector": model_cfg.get("connector"),
                "strip_thinking_enabled": strip_thinking_enabled,
                "strip_thinking_changed_count": strip_thinking_changed,
                "raw_has_think_count": raw_has_think,
                "grade": grade,
                "headline": headline,
                "notes": notes,
                "results_path": str(out_path),
                "details_path": str(details_path),
                "details_rel": details_rel,
            }
        )

        print(f"Wrote {out_path}")
        print(
            f"{interrogator.model_name}: exact={strict_hits}/{len(results)} "
            f"halluc={hallucinations}/{len(results)} avg_wer={avg_wer:.3f}"
        )

    prompts_blob = json.dumps(cfg.get("prompts", {}), sort_keys=True, ensure_ascii=False)
    prompt_hash = hashlib.sha256(prompts_blob.encode("utf-8")).hexdigest()[:12]
    repro = {
        "tool_version": __version__,
        "scorer_config": asdict(ScoreConfig()),
        "python": platform.python_version(),
        "seed": sample_cfg.seed,
        "prompt_mode": prompt_mode,
        "prompt_hash": prompt_hash,
        "config_path": args.config,
        "command": f"bible-eval run --config {args.config}",
        **_git_state(),
    }

    run_summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": version_key,
        "prompt_mode": prompt_mode,
        "sample": asdict(sample_cfg),
        "schema_version": 2,
        "repro": repro,
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
        rels = [m.get("details_rel"), (m.get("abstention") or {}).get("details_rel")]
        for rel in rels:
            if not rel:
                continue
            src = Path("results") / rel
            if not src.exists():
                continue
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
            rels = [m.get("details_rel"), (m.get("abstention") or {}).get("details_rel")]
            for rel in rels:
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
