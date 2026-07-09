import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from boundary_candidates import generate_boundary_candidates
from checkpoint import checkpoint_to_rows, load_checkpoint, should_skip, upsert_checkpoint
from config import ESTIMATED_TOKENS_PER_CALL, MODEL_NAME, N_PER_CASE, ROOT, T, TEMPERATURE
from data_gen import generate_instance
from prompts import build_sanity5c_prompt
from sanity5_parser import parse_boundary_response
from visualize import render_overlay
from vlm_client import VLMClient, VLMResponse


# Sanity-5c is a second ablation of Sanity-5: same L0-L3/R0-R3 candidates and response
# schema, but nothing is drawn on the image at all (plain two-channel overlay, like
# Sanity-1/3). Candidate positions are given only as numbers in the prompt text
# ("L0=96, L1=126, ..."). This isolates visual grounding from symbolic reasoning: can
# the model judge the break location from the raw shapes and then map that judgment
# onto the closest listed number, without any drawn markers to lean on?
SANITY5C_ROOT = ROOT / "results" / "sanity5c" / "runs"

PRIMARY_CASES = ["C1", "C2", "C3"]
REFERENCE_CASES = ["C4", "C5"]
ALL_CASES = PRIMARY_CASES + REFERENCE_CASES

RAW_COLUMNS = [
    "case_id", "case_type", "group", "break_start", "break_end",
    "left_L0", "left_L1", "left_L2", "left_L3",
    "right_R0", "right_R1", "right_R2", "right_R3",
    "gold_left", "gold_right", "model_left", "model_right",
    "left_correct", "right_correct", "both_correct",
    "left_error_steps", "right_error_steps",
    "model_confidence", "model_reason", "parse_status",
]


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "images": run_dir / "images",
        "logs": run_dir / "logs",
        "raw": run_dir / "raw",
        "diagnosis": run_dir / "diagnosis",
        "checkpoint": run_dir / "checkpoint.json",
        "infer_log": run_dir / "logs" / "sanity5c_inferences.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
        "raw_csv": run_dir / "raw" / "sanity5c_raw.csv",
        "summary": run_dir / "sanity5c_summary.json",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mock_call(gold_left: str, gold_right: str) -> VLMResponse:
    raw = json.dumps({
        "left_option": gold_left,
        "right_option": gold_right,
        "reason": "Mock boundary-selection response for pipeline validation.",
        "confidence": 0.99,
    })
    return VLMResponse("OK", raw_response=raw, input_tokens=0, output_tokens=0, total_tokens=0, latency_seconds=0.0)


def _block_metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {
            "n": 0, "left_accuracy": None, "right_accuracy": None, "both_accuracy": None,
            "left_error_mae": None, "right_error_mae": None,
        }
    return {
        "n": int(n),
        "left_accuracy": float(sub["left_correct"].mean()),
        "right_accuracy": float(sub["right_correct"].mean()),
        "both_accuracy": float(sub["both_correct"].mean()),
        "left_error_mae": float(sub["left_error_steps"].mean()),
        "right_error_mae": float(sub["right_error_steps"].mean()),
    }


def _build_summary(df: pd.DataFrame) -> dict:
    scored = df[df["parse_status"] == "OK"].copy()
    per_case = {str(k): _block_metrics(v) for k, v in scored.groupby("case_type")}
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_per_case": N_PER_CASE,
        "T": T,
        "primary_cases": PRIMARY_CASES,
        "reference_cases": REFERENCE_CASES,
        "parse_status_counts": {str(k): int(v) for k, v in df["parse_status"].value_counts(dropna=False).to_dict().items()},
        "overall": _block_metrics(scored),
        "primary_cases_only": _block_metrics(scored[scored["group"] == "primary"]),
        "reference_cases_only": _block_metrics(scored[scored["group"] == "reference"]),
        "per_case": per_case,
        "left_option_pick_distribution": {str(k): int(v) for k, v in scored["model_left"].value_counts().to_dict().items()},
        "right_option_pick_distribution": {str(k): int(v) for k, v in scored["model_right"].value_counts().to_dict().items()},
        "gold_left_distribution": {str(k): int(v) for k, v in scored["gold_left"].value_counts().to_dict().items()},
        "gold_right_distribution": {str(k): int(v) for k, v in scored["gold_right"].value_counts().to_dict().items()},
    }


def _copy_example(paths: dict[str, Path], row: pd.Series, prefix: str) -> None:
    src = paths["images"] / f"{row['case_id']}.png"
    if src.exists():
        shutil.copyfile(src, paths["diagnosis"] / f"{prefix}_{row['case_id']}.png")


def _diagnosis(df: pd.DataFrame, paths: dict[str, Path], summary: dict) -> None:
    paths["diagnosis"].mkdir(parents=True, exist_ok=True)
    scored = df[df["parse_status"] == "OK"].copy()

    for group in ["primary", "reference"]:
        sub = scored[scored["group"] == group]
        if sub.empty:
            continue
        success = sub[sub["both_correct"]]
        failure = sub[~sub["both_correct"]]
        if not success.empty:
            _copy_example(paths, success.sort_values("model_confidence", ascending=False).iloc[0], f"success_{group}")
        if not failure.empty:
            _copy_example(paths, failure.sort_values("left_error_steps", ascending=False).iloc[0], f"failure_{group}")

    lines = ["# Sanity-5c Diagnosis (Constrained Boundary Selection, text-only candidates)", ""]
    lines.append(
        "Second ablation of Sanity-5: the image has no drawn markers at all (plain "
        "overlay); L0-L3/R0-R3 positions are given only as numbers in the prompt text. "
        "Tests visual judgment + symbolic mapping without any drawn visual anchor.\n"
    )
    lines.append(f"Overall: {summary['overall']}\n")
    lines.append(f"Primary (C1/C2/C3) only: {summary['primary_cases_only']}\n")
    lines.append(f"Reference (C4/C5) only: {summary['reference_cases_only']}\n")
    lines.append(f"Model's left_option pick distribution: {summary['left_option_pick_distribution']}\n")
    lines.append(f"Model's right_option pick distribution: {summary['right_option_pick_distribution']}\n")
    for case, metric in summary["per_case"].items():
        group = "primary" if case in PRIMARY_CASES else "reference"
        lines.append(f"## {case} ({group})\nMetrics: {metric}\n")
    (paths["diagnosis"] / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_dir = SANITY5C_ROOT / args.resume
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume missing Sanity-5c run: {args.resume}")
        return args.resume, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, SANITY5C_ROOT / run_id


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
    print("\nSanity-5c Results")
    print(f"Parse status: {summary['parse_status_counts']}")
    print(f"Overall: {summary['overall']}")
    print(f"Primary (C1/C2/C3): {summary['primary_cases_only']}")
    print(f"Reference (C4/C5): {summary['reference_cases_only']}")
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

    total_calls = len(ALL_CASES) * N_PER_CASE
    print(f"Sanity-5c Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Planned calls: {total_calls}; rough token budget: ~{total_calls * ESTIMATED_TOKENS_PER_CALL:,} tokens")
    if not args.dry_run and not args.yes:
        ans = input("Proceed with real OpenAI API calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = None if args.dry_run else VLMClient(MODEL_NAME)

    for case_type in ALL_CASES:
        group = "primary" if case_type in PRIMARY_CASES else "reference"
        for index in range(N_PER_CASE):
            case_id, _, a, b, meta = generate_instance(case_type, index, T)
            image_path = paths["images"] / f"{case_id}.png"

            cand_seed = 500_000 + 10_000 * int(case_type[1:]) + index
            cands = generate_boundary_candidates(
                a, b, meta["break_start"], meta["break_end"], seed=cand_seed, t=T,
            )
            img_b64 = render_overlay(a, b, image_path)

            if should_skip(checkpoint.get(case_id)):
                print(f"[SKIP] {case_id}")
                continue

            prompt = build_sanity5c_prompt(cands["left"], cands["right"])
            if args.dry_run:
                resp = _mock_call(cands["gold_left"], cands["gold_right"])
            else:
                resp = client.call(prompt, img_b64)

            base_row = {
                "case_type": case_type,
                "group": group,
                "break_start": meta["break_start"],
                "break_end": meta["break_end"],
                "left_L0": cands["left"]["L0"], "left_L1": cands["left"]["L1"],
                "left_L2": cands["left"]["L2"], "left_L3": cands["left"]["L3"],
                "right_R0": cands["right"]["R0"], "right_R1": cands["right"]["R1"],
                "right_R2": cands["right"]["R2"], "right_R3": cands["right"]["R3"],
                "gold_left": cands["gold_left"],
                "gold_right": cands["gold_right"],
            }

            if resp.status == "API_ERROR":
                parse_status = "API_ERROR"
                row = {
                    **base_row,
                    "model_left": None, "model_right": None,
                    "left_correct": None, "right_correct": None, "both_correct": None,
                    "left_error_steps": None, "right_error_steps": None,
                    "model_confidence": None,
                    "model_reason": resp.error or "",
                    "parse_status": parse_status,
                }
            else:
                parsed = parse_boundary_response(resp.raw_response)
                parse_status = parsed.status
                if parsed.status == "PARSE_ERROR":
                    _append_jsonl(paths["parse_failures"], {
                        "case_id": case_id,
                        "failure_reason": parsed.failure_reason,
                        "raw_response": resp.raw_response,
                    })
                    row = {
                        **base_row,
                        "model_left": None, "model_right": None,
                        "left_correct": None, "right_correct": None, "both_correct": None,
                        "left_error_steps": None, "right_error_steps": None,
                        "model_confidence": None,
                        "model_reason": parsed.failure_reason or "",
                        "parse_status": parse_status,
                    }
                else:
                    left_correct = parsed.left_option == cands["gold_left"]
                    right_correct = parsed.right_option == cands["gold_right"]
                    row = {
                        **base_row,
                        "model_left": parsed.left_option,
                        "model_right": parsed.right_option,
                        "left_correct": bool(left_correct),
                        "right_correct": bool(right_correct),
                        "both_correct": bool(left_correct and right_correct),
                        "left_error_steps": abs(cands["left"][parsed.left_option] - meta["break_start"]),
                        "right_error_steps": abs(cands["right"][parsed.right_option] - meta["break_end"]),
                        "model_confidence": parsed.confidence,
                        "model_reason": parsed.reason,
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
                "model_left": row["model_left"],
                "model_right": row["model_right"],
            })
            upsert_checkpoint(paths["checkpoint"], case_id, row)
            checkpoint[case_id] = row
            print(
                f"[{parse_status}] {case_id}: gold=({cands['gold_left']},{cands['gold_right']}) "
                f"pred=({row['model_left']},{row['model_right']})"
            )

    summary = _write_outputs(paths)
    _print_summary(summary, paths["summary"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
