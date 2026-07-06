import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from checkpoint import checkpoint_to_rows, load_checkpoint, should_skip, upsert_checkpoint
from config import CASES, ESTIMATED_TOKENS_PER_CALL, MODEL_NAME, N_PER_CASE, ROOT, T, TEMPERATURE
from data_gen import generate_all
from metrics import classification_metrics, interval_iou, localization_metrics
from prompts import SANITY3_PROMPT
from sanity3_parser import parse_localization_response
from visualize import render_overlay
from vlm_client import VLMClient, VLMResponse


SANITY3_ROOT = ROOT / "results" / "sanity3" / "runs"

RAW_COLUMNS = [
    "case_id", "case_type", "ground_truth_label", "expected_break_start", "expected_break_end",
    "model_answer", "predicted_break_start", "predicted_break_end", "interval_iou",
    "model_confidence", "model_reason", "parse_status",
]


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "images": run_dir / "images",
        "logs": run_dir / "logs",
        "raw": run_dir / "raw",
        "diagnosis": run_dir / "diagnosis",
        "checkpoint": run_dir / "checkpoint.json",
        "infer_log": run_dir / "logs" / "sanity3_inferences.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
        "raw_csv": run_dir / "raw" / "sanity3_raw.csv",
        "summary": run_dir / "sanity3_summary.json",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mock_call(label: str, break_start: int | None, break_end: int | None) -> VLMResponse:
    raw = json.dumps({
        "answer": label,
        "break_start": break_start,
        "break_end": break_end,
        "reason": "Mock localization response for pipeline validation.",
        "confidence": 0.99,
    })
    return VLMResponse("OK", raw_response=raw, input_tokens=0, output_tokens=0, total_tokens=0, latency_seconds=0.0)


def _row_iou(expected_start, expected_end, predicted_start, predicted_end) -> float | None:
    return interval_iou(expected_start, expected_end, predicted_start, predicted_end)


def _build_summary(df: pd.DataFrame) -> dict:
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_per_case": N_PER_CASE,
        "T": T,
        "parse_status_counts": {str(k): int(v) for k, v in df["parse_status"].value_counts(dropna=False).to_dict().items()},
        "classification_metrics": classification_metrics(df),
        "classification_by_case": classification_metrics(df, "case_type"),
        "localization_metrics": localization_metrics(df),
    }


def _copy_example(paths: dict[str, Path], row: pd.Series, prefix: str) -> None:
    src = paths["images"] / f"{row['case_id']}.png"
    if src.exists():
        shutil.copyfile(src, paths["diagnosis"] / f"{prefix}_{row['case_id']}.png")


def _diagnosis(df: pd.DataFrame, paths: dict[str, Path], summary: dict) -> None:
    paths["diagnosis"].mkdir(parents=True, exist_ok=True)
    ok = df[df["parse_status"] == "OK"].copy()
    broken = ok[ok["ground_truth_label"] == "broken"].copy()
    maintained = ok[ok["ground_truth_label"] == "maintained"].copy()

    if not broken.empty:
        best = broken.sort_values("interval_iou", ascending=False).head(1)
        worst = broken.sort_values("interval_iou", ascending=True).head(1)
        for _, row in best.iterrows():
            _copy_example(paths, row, "success_localization")
        for _, row in worst.iterrows():
            _copy_example(paths, row, "failure_localization")

    false_loc = maintained[
        maintained["predicted_break_start"].notna() | maintained["predicted_break_end"].notna()
    ]
    for _, row in false_loc.head(1).iterrows():
        _copy_example(paths, row, "false_localization")

    lines = ["# Sanity-3 Diagnosis", ""]
    lines.append(
        "Sanity-3 tests whether the VLM can localize the relationship-break interval, "
        "not just classify the plot as maintained or broken.\n"
    )
    loc = summary["localization_metrics"]
    lines.append(
        "Overall localization: "
        f"IoU mean={loc['interval_iou_mean']}, IoU median={loc['interval_iou_median']}, "
        f"hit@0.5={loc['hit_iou_0.5']}, false localization={loc['false_localization_rate']}.\n"
    )
    for case, sub in ok.groupby("case_type"):
        metric = summary["classification_by_case"].get(case, {})
        case_loc = localization_metrics(sub)
        lines.append(
            f"## {case}\n"
            f"Accuracy: {metric.get('accuracy')}. "
            f"IoU mean: {case_loc.get('interval_iou_mean')}. "
            f"False localization rate: {case_loc.get('false_localization_rate')}.\n"
        )
    (paths["diagnosis"] / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_dir = SANITY3_ROOT / args.resume
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume missing Sanity-3 run: {args.resume}")
        return args.resume, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, SANITY3_ROOT / run_id


def _write_outputs(paths: dict[str, Path]) -> dict:
    rows = checkpoint_to_rows(load_checkpoint(paths["checkpoint"]))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No checkpoint rows produced.")
    paths["raw"].mkdir(parents=True, exist_ok=True)
    df[RAW_COLUMNS].to_csv(paths["raw_csv"], index=False)
    summary = _build_summary(df)
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _diagnosis(df, paths, summary)
    return summary


def _print_summary(summary: dict, summary_path: Path) -> None:
    print("\nSanity-3 Results")
    print(f"Parse status: {summary['parse_status_counts']}")
    print(f"Classification: {summary['classification_metrics']}")
    print(f"Localization: {summary['localization_metrics']}")
    print(f"\nSaved summary: {summary_path}")


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

    total_calls = len(CASES) * N_PER_CASE
    print(f"Sanity-3 Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Planned calls: {total_calls}; rough token budget: ~{total_calls * ESTIMATED_TOKENS_PER_CALL:,} tokens")
    if not args.dry_run and not args.yes:
        ans = input("Proceed with real OpenAI API calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = None if args.dry_run else VLMClient(MODEL_NAME)

    for case_id, case_type, a, b, meta in generate_all(N_PER_CASE, T, CASES):
        image_path = paths["images"] / f"{case_id}.png"
        img_b64 = render_overlay(a, b, image_path)

        if should_skip(checkpoint.get(case_id)):
            print(f"[SKIP] {case_id}")
            continue

        if args.dry_run:
            resp = _mock_call(meta["label"], meta["break_start"], meta["break_end"])
        else:
            resp = client.call(SANITY3_PROMPT, img_b64)

        if resp.status == "API_ERROR":
            parse_status = "API_ERROR"
            row = {
                "status": parse_status,
                "case_type": case_type,
                "ground_truth_label": meta["label"],
                "expected_break_start": meta["break_start"],
                "expected_break_end": meta["break_end"],
                "model_answer": None,
                "predicted_break_start": None,
                "predicted_break_end": None,
                "interval_iou": None,
                "model_confidence": None,
                "model_reason": resp.error or "",
                "parse_status": parse_status,
            }
        else:
            parsed = parse_localization_response(resp.raw_response, T)
            parse_status = parsed.status
            if parsed.status == "PARSE_ERROR":
                _append_jsonl(paths["parse_failures"], {
                    "case_id": case_id,
                    "failure_reason": parsed.failure_reason,
                    "raw_response": resp.raw_response,
                })
            iou = None
            if parsed.status == "OK":
                iou = _row_iou(meta["break_start"], meta["break_end"], parsed.break_start, parsed.break_end)
            row = {
                "status": parse_status,
                "case_type": case_type,
                "ground_truth_label": meta["label"],
                "expected_break_start": meta["break_start"],
                "expected_break_end": meta["break_end"],
                "model_answer": parsed.answer,
                "predicted_break_start": parsed.break_start,
                "predicted_break_end": parsed.break_end,
                "interval_iou": iou,
                "model_confidence": parsed.confidence,
                "model_reason": parsed.reason or parsed.failure_reason or "",
                "parse_status": parse_status,
            }

        _append_jsonl(paths["infer_log"], {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "model_name": MODEL_NAME,
            "temperature": TEMPERATURE,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "total_tokens": resp.total_tokens,
            "latency_seconds": resp.latency_seconds,
            "raw_response": resp.raw_response,
            "parse_status": parse_status,
            "predicted_answer": row["model_answer"],
            "predicted_break_start": row["predicted_break_start"],
            "predicted_break_end": row["predicted_break_end"],
        })
        upsert_checkpoint(paths["checkpoint"], case_id, row)
        checkpoint[case_id] = row
        print(
            f"[{parse_status}] {case_id}: gt={meta['label']} pred={row['model_answer']} "
            f"interval=({row['predicted_break_start']}, {row['predicted_break_end']}) iou={row['interval_iou']}"
        )

    summary = _write_outputs(paths)
    _print_summary(summary, paths["summary"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
