import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from checkpoint import checkpoint_to_rows, load_checkpoint, should_skip, upsert_checkpoint
from config import CASES, ESTIMATED_TOKENS_PER_CALL, MODEL_NAME, N_PER_CASE, RESULTS_ROOT, T, TEMPERATURE
from data_gen import generate_all
from metrics import classification_metrics, confidence_analysis
from parser import parse_response
from prompts import SANITY1_PROMPT
from visualize import render_overlay
from vlm_client import MockVLMClient, VLMClient


RAW_COLUMNS = [
    "case_id", "case_type", "ground_truth_label", "model_answer",
    "model_confidence", "model_reason", "parse_status",
]


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "images": run_dir / "images",
        "logs": run_dir / "logs",
        "raw": run_dir / "raw",
        "diagnosis": run_dir / "diagnosis",
        "checkpoint": run_dir / "checkpoint.json",
        "infer_log": run_dir / "logs" / "sanity1_inferences.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
        "raw_csv": run_dir / "raw" / "sanity1_raw.csv",
        "failure_csv": run_dir / "raw" / "failure_log.csv",
        "summary": run_dir / "sanity1_summary.json",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _case_verdict(case_type: str, metric: dict) -> str:
    acc = metric.get("accuracy")
    f1 = metric.get("f1_broken")
    if acc is None and f1 is None:
        return "NO_DATA"
    if case_type in {"C4", "C5"}:
        return "BOUNDARY (no threshold applied)"
    if case_type in {"C0", "C6"}:
        return "WARNING" if acc is not None and acc < 0.90 else "PASS"
    value = max(v for v in [acc, f1] if v is not None) if (acc is not None or f1 is not None) else 0.0
    return "PASS" if value >= 0.90 else "FAIL"


def _build_summary(df: pd.DataFrame) -> dict:
    by_case = classification_metrics(df, "case_type")
    conf_case = confidence_analysis(df, "case_type")
    status_counts = df["parse_status"].value_counts(dropna=False).to_dict()
    verdicts = {case: _case_verdict(case, by_case.get(case, {})) for case in CASES}
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_per_case": N_PER_CASE,
        "T": T,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "overall_metrics": classification_metrics(df),
        "overall_confidence": confidence_analysis(df),
        "per_case_metrics": by_case,
        "per_case_confidence": conf_case,
        "verdicts": verdicts,
    }


def _write_tables(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    paths["raw"].mkdir(parents=True, exist_ok=True)
    df[RAW_COLUMNS].to_csv(paths["raw_csv"], index=False)
    failures = df[
        (df["parse_status"] == "OK")
        & (df["model_answer"].notna())
        & (df["ground_truth_label"] != df["model_answer"])
    ]
    failures.rename(columns={
        "ground_truth_label": "ground_truth",
        "model_confidence": "model_confidence",
        "model_reason": "model_reason",
    })[["case_id", "case_type", "ground_truth", "model_answer", "model_reason", "model_confidence"]].to_csv(
        paths["failure_csv"], index=False
    )


def _make_diagnosis(df: pd.DataFrame, paths: dict[str, Path], summary: dict) -> None:
    paths["diagnosis"].mkdir(parents=True, exist_ok=True)
    failures = pd.DataFrame()
    if paths["failure_csv"].exists():
        failures = pd.read_csv(paths["failure_csv"])
    for _, row in failures.iterrows():
        src = paths["images"] / f"{row['case_id']}.png"
        if src.exists():
            shutil.copyfile(src, paths["diagnosis"] / f"failure_{row['case_id']}.png")

    scored = df[(df["parse_status"] == "OK") & (df["ground_truth_label"] == df["model_answer"])].copy()
    for case_type in CASES:
        sub = scored[scored["case_type"] == case_type]
        if sub.empty:
            continue
        best = sub.sort_values("model_confidence", ascending=False).iloc[0]
        src = paths["images"] / f"{best['case_id']}.png"
        if src.exists():
            shutil.copyfile(src, paths["diagnosis"] / f"success_{case_type}.png")

    lines = ["# Sanity-1 Diagnosis", ""]
    manip = {"C1": "amplitude jump", "C2": "flatline", "C3": "phase flip"}
    for case_type in CASES:
        metric = summary["per_case_metrics"].get(case_type, {})
        acc = metric.get("accuracy")
        verdict = summary["verdicts"].get(case_type)
        sub_fail = failures[failures["case_type"] == case_type] if not failures.empty else pd.DataFrame()
        note = ""
        if case_type in manip:
            successes = scored[scored["case_type"] == case_type]
            reasons = " ".join(str(x).lower() for x in successes["model_reason"].head(5).tolist())
            note = f" Success reasons were checked for the expected {manip[case_type]} cue; sampled reasons: {reasons[:300] or 'none'}."
        root = ""
        if verdict in {"FAIL", "WARNING"} and not sub_fail.empty:
            reason_text = " ".join(str(x).lower() for x in sub_fail["model_reason"].head(5).tolist())
            root = (
                " Root-cause note: failures appear to require manual review; the preserved reasons suggest "
                + ("a reasoning failure." if any(w in reason_text for w in ["sync", "relationship", "similar"]) else "a possible perception failure.")
            )
        lines.append(
            f"## {case_type}\nAccuracy: {acc if acc is not None else 'N/A'}. Verdict: {verdict}.{note}{root}\n"
        )
    (paths["diagnosis"] / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _print_summary(summary: dict) -> None:
    print("\nSanity-1 Results")
    print("case  n   acc     f1_broken  confidence  verdict")
    print("----  --  ------  ---------  ----------  -----------------------------")
    for case in CASES:
        m = summary["per_case_metrics"].get(case, {})
        c = summary["per_case_confidence"].get(case, {})
        def fmt(v):
            return "N/A" if v is None else f"{v:.3f}"
        print(
            f"{case:<4}  {m.get('n', 0):>2}  {fmt(m.get('accuracy')):>6}  "
            f"{fmt(m.get('f1_broken')):>9}  {fmt(c.get('mean_confidence')):>10}  "
            f"{summary['verdicts'].get(case)}"
        )
    overall = summary["overall_metrics"]
    print(f"\nOverall accuracy: {overall.get('accuracy')}")
    print(f"Status counts: {summary['status_counts']}")
    c0_acc = summary["per_case_metrics"].get("C0", {}).get("accuracy")
    c6_acc = summary["per_case_metrics"].get("C6", {}).get("accuracy")
    if c0_acc is not None and c0_acc < 0.90:
        print("CRITICAL WARNING: C0 negative control below 90% accuracy.")
    if c6_acc is not None and c6_acc < 0.90:
        print("CRITICAL WARNING: C6 noise negative control below 90% accuracy.")


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_id = args.resume
        run_dir = RESULTS_ROOT / run_id
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume: missing run folder or checkpoint for {run_id}")
        return run_id, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, RESULTS_ROOT / run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use mocked VLM responses; no API cost.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation for real API run.")
    parser.add_argument("--resume", default=None, help="Resume an existing run_id.")
    args = parser.parse_args()

    run_id, run_dir = _run_dir(args)
    paths = _paths(run_dir)
    for key in ["images", "logs", "raw", "diagnosis"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    total_calls = len(CASES) * N_PER_CASE
    print(f"Planned calls: {total_calls}; rough token budget: ~{total_calls * ESTIMATED_TOKENS_PER_CALL:,} tokens")
    if not args.dry_run and not args.yes:
        ans = input("Proceed with real OpenAI API calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = MockVLMClient() if args.dry_run else VLMClient(MODEL_NAME)
    instances = generate_all(N_PER_CASE, T, CASES)

    for case_id, case_type, a, b, meta in instances:
        image_path = paths["images"] / f"{case_id}.png"
        img_b64 = render_overlay(a, b, image_path)

        if should_skip(checkpoint.get(case_id)):
            print(f"[SKIP] {case_id}")
            continue

        if args.dry_run:
            vlm_resp = client.call(SANITY1_PROMPT, img_b64, answer=meta["label"])
        else:
            vlm_resp = client.call(SANITY1_PROMPT, img_b64)

        if vlm_resp.status == "API_ERROR":
            parsed = None
            parse_status = "API_ERROR"
            row = {
                "status": "API_ERROR",
                "case_type": case_type,
                "ground_truth_label": meta["label"],
                "model_answer": None,
                "model_confidence": None,
                "model_reason": vlm_resp.error or "",
                "parse_status": parse_status,
            }
        else:
            parsed = parse_response(vlm_resp.raw_response)
            parse_status = parsed.status
            if parsed.status == "PARSE_ERROR":
                _append_jsonl(paths["parse_failures"], {
                    "case_id": case_id,
                    "failure_reason": parsed.failure_reason,
                    "raw_response": vlm_resp.raw_response,
                })
            row = {
                "status": parsed.status,
                "case_type": case_type,
                "ground_truth_label": meta["label"],
                "model_answer": parsed.answer,
                "model_confidence": parsed.confidence,
                "model_reason": parsed.reason or parsed.failure_reason or "",
                "parse_status": parse_status,
            }

        _append_jsonl(paths["infer_log"], {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "model_name": MODEL_NAME,
            "temperature": TEMPERATURE,
            "input_tokens": vlm_resp.input_tokens,
            "output_tokens": vlm_resp.output_tokens,
            "total_tokens": vlm_resp.total_tokens,
            "latency_seconds": vlm_resp.latency_seconds,
            "raw_response": vlm_resp.raw_response,
            "parsed_answer": row["model_answer"],
            "parsed_confidence": row["model_confidence"],
            "parse_status": parse_status,
        })
        upsert_checkpoint(paths["checkpoint"], case_id, row)
        checkpoint[case_id] = row
        print(f"[{parse_status}] {case_id}: gt={meta['label']} pred={row['model_answer']}")

    final_rows = checkpoint_to_rows(load_checkpoint(paths["checkpoint"]))
    df = pd.DataFrame(final_rows)
    if df.empty:
        raise SystemExit("No checkpoint rows produced.")
    _write_tables(df, paths)
    summary = _build_summary(df)
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _make_diagnosis(df, paths, summary)
    _print_summary(summary)
    print(f"\nSaved summary: {paths['summary']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
