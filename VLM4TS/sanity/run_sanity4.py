import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from checkpoint import checkpoint_to_rows, load_checkpoint, should_skip, upsert_checkpoint
from config import ESTIMATED_TOKENS_PER_CALL, MODEL_NAME, N_PER_CASE, ROOT, T, TEMPERATURE
from data_gen import GENERATORS
from prompts import SANITY4_PROMPT
from sanity4_parser import parse_candidate_response
from visualize import render_candidate_overlay
from vlm_client import VLMClient, VLMResponse


SANITY4_ROOT = ROOT / "results" / "sanity4" / "runs"
SCENARIOS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]

RAW_COLUMNS = [
    "case_id", "scenario", "source_case", "candidate_start", "candidate_end",
    "ground_truth_label", "model_answer", "model_confidence", "model_reason", "parse_status",
]


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "images": run_dir / "images",
        "logs": run_dir / "logs",
        "raw": run_dir / "raw",
        "diagnosis": run_dir / "diagnosis",
        "checkpoint": run_dir / "checkpoint.json",
        "infer_log": run_dir / "logs" / "sanity4_inferences.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
        "raw_csv": run_dir / "raw" / "sanity4_raw.csv",
        "failure_csv": run_dir / "raw" / "failure_log.csv",
        "summary": run_dir / "sanity4_summary.json",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scenario(index: int, scenario: str):
    mapping = {
        "V0": ("C0", 100, 200, "invalid", "maintained negative control"),
        "V1": ("C1", 100, 200, "valid", "amplitude break candidate"),
        "V2": ("C2", 100, 200, "valid", "flatline break candidate"),
        "V3": ("C3", 0, 100, "invalid", "false candidate before phase flip"),
        "V4": ("C4", 150, 250, "valid", "late sustained phase drift candidate"),
        "V5": ("C5", 100, 200, "valid", "subtle frequency drift candidate"),
        "V6": ("C6", 100, 200, "invalid", "noisy maintained negative control"),
    }
    source_case, cand_start, cand_end, verdict, description = mapping[scenario]
    seed = 10_000 * int(source_case[1:]) + index
    a, b, meta = GENERATORS[source_case](seed=seed, t=T)
    return {
        "case_id": f"{scenario}_{index:03d}",
        "scenario": scenario,
        "source_case": source_case,
        "a": a,
        "b": b,
        "candidate_start": cand_start,
        "candidate_end": cand_end,
        "ground_truth_label": verdict,
        "source_break_start": meta["break_start"],
        "source_break_end": meta["break_end"],
        "description": description,
    }


def _generate_all():
    for scenario in SCENARIOS:
        for index in range(N_PER_CASE):
            yield _scenario(index, scenario)


def _mock_call(verdict: str) -> VLMResponse:
    raw = json.dumps({
        "verdict": verdict,
        "reason": "Mock candidate-verification response for pipeline validation.",
        "confidence": 0.99,
    })
    return VLMResponse("OK", raw_response=raw, input_tokens=0, output_tokens=0, total_tokens=0, latency_seconds=0.0)


def _candidate_metrics(df: pd.DataFrame, group_col: str | None = None) -> dict:
    def one(sub: pd.DataFrame) -> dict:
        scored = sub[(sub["parse_status"] == "OK") & sub["model_answer"].notna()].copy()
        if scored.empty:
            return {"n": 0, "accuracy": None, "precision_valid": None, "recall_valid": None, "f1_valid": None}
        correct = scored["ground_truth_label"] == scored["model_answer"]
        tp = int(((scored["ground_truth_label"] == "valid") & (scored["model_answer"] == "valid")).sum())
        fp = int(((scored["ground_truth_label"] == "invalid") & (scored["model_answer"] == "valid")).sum())
        fn = int(((scored["ground_truth_label"] == "valid") & (scored["model_answer"] == "invalid")).sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {
            "n": int(len(scored)),
            "accuracy": float(correct.mean()),
            "precision_valid": float(precision),
            "recall_valid": float(recall),
            "f1_valid": float(f1),
        }

    if group_col:
        return {str(k): one(v) for k, v in df.groupby(group_col)}
    return one(df)


def _build_summary(df: pd.DataFrame) -> dict:
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_per_scenario": N_PER_CASE,
        "T": T,
        "parse_status_counts": {str(k): int(v) for k, v in df["parse_status"].value_counts(dropna=False).to_dict().items()},
        "overall_metrics": _candidate_metrics(df),
        "per_scenario_metrics": _candidate_metrics(df, "scenario"),
        "per_source_case_metrics": _candidate_metrics(df, "source_case"),
    }


def _write_tables(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    paths["raw"].mkdir(parents=True, exist_ok=True)
    df[RAW_COLUMNS].to_csv(paths["raw_csv"], index=False)
    failures = df[
        (df["parse_status"] == "OK")
        & (df["model_answer"].notna())
        & (df["ground_truth_label"] != df["model_answer"])
    ]
    failures[RAW_COLUMNS].to_csv(paths["failure_csv"], index=False)


def _copy_example(paths: dict[str, Path], row: pd.Series, prefix: str) -> None:
    src = paths["images"] / f"{row['case_id']}.png"
    if src.exists():
        shutil.copyfile(src, paths["diagnosis"] / f"{prefix}_{row['case_id']}.png")


def _diagnosis(df: pd.DataFrame, paths: dict[str, Path], summary: dict) -> None:
    paths["diagnosis"].mkdir(parents=True, exist_ok=True)
    ok = df[df["parse_status"] == "OK"].copy()
    ok["correct"] = ok["ground_truth_label"] == ok["model_answer"]
    for verdict in ["valid", "invalid"]:
        sub = ok[(ok["ground_truth_label"] == verdict) & ok["correct"]]
        if not sub.empty:
            _copy_example(paths, sub.sort_values("model_confidence", ascending=False).iloc[0], f"success_{verdict}")
    failures = ok[~ok["correct"]]
    for _, row in failures.head(5).iterrows():
        _copy_example(paths, row, "failure")

    lines = ["# Sanity-4 Diagnosis", ""]
    lines.append(
        "Sanity-4 tests whether the VLM can verify a highlighted candidate interval, "
        "including rejecting false candidates when a break occurs elsewhere.\n"
    )
    lines.append(f"Overall metrics: {summary['overall_metrics']}\n")
    for scenario, metric in summary["per_scenario_metrics"].items():
        sub = ok[ok["scenario"] == scenario]
        desc = str(sub["description"].iloc[0]) if not sub.empty else ""
        lines.append(f"## {scenario}\n{desc}. Metrics: {metric}\n")
    (paths["diagnosis"] / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_dir = SANITY4_ROOT / args.resume
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume missing Sanity-4 run: {args.resume}")
        return args.resume, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, SANITY4_ROOT / run_id


def _write_outputs(paths: dict[str, Path]) -> dict:
    rows = checkpoint_to_rows(load_checkpoint(paths["checkpoint"]))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No checkpoint rows produced.")
    _write_tables(df, paths)
    summary = _build_summary(df)
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _diagnosis(df, paths, summary)
    return summary


def _print_summary(summary: dict, summary_path: Path) -> None:
    print("\nSanity-4 Results")
    print(f"Parse status: {summary['parse_status_counts']}")
    print(f"Overall: {summary['overall_metrics']}")
    for scenario, metric in summary["per_scenario_metrics"].items():
        print(f"{scenario}: acc={metric['accuracy']}, f1_valid={metric['f1_valid']}")
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

    total_calls = len(SCENARIOS) * N_PER_CASE
    print(f"Sanity-4 Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Planned calls: {total_calls}; rough token budget: ~{total_calls * ESTIMATED_TOKENS_PER_CALL:,} tokens")
    if not args.dry_run and not args.yes:
        ans = input("Proceed with real OpenAI API calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = None if args.dry_run else VLMClient(MODEL_NAME)

    for case in _generate_all():
        case_id = case["case_id"]
        image_path = paths["images"] / f"{case_id}.png"
        img_b64 = render_candidate_overlay(
            case["a"], case["b"], case["candidate_start"], case["candidate_end"], image_path
        )

        if should_skip(checkpoint.get(case_id)):
            print(f"[SKIP] {case_id}")
            continue

        if args.dry_run:
            resp = _mock_call(case["ground_truth_label"])
        else:
            resp = client.call(SANITY4_PROMPT, img_b64)

        if resp.status == "API_ERROR":
            parse_status = "API_ERROR"
            row = {
                "status": parse_status,
                "scenario": case["scenario"],
                "source_case": case["source_case"],
                "candidate_start": case["candidate_start"],
                "candidate_end": case["candidate_end"],
                "source_break_start": case["source_break_start"],
                "source_break_end": case["source_break_end"],
                "description": case["description"],
                "ground_truth_label": case["ground_truth_label"],
                "model_answer": None,
                "model_confidence": None,
                "model_reason": resp.error or "",
                "parse_status": parse_status,
            }
        else:
            parsed = parse_candidate_response(resp.raw_response)
            parse_status = parsed.status
            if parsed.status == "PARSE_ERROR":
                _append_jsonl(paths["parse_failures"], {
                    "case_id": case_id,
                    "failure_reason": parsed.failure_reason,
                    "raw_response": resp.raw_response,
                })
            row = {
                "status": parse_status,
                "scenario": case["scenario"],
                "source_case": case["source_case"],
                "candidate_start": case["candidate_start"],
                "candidate_end": case["candidate_end"],
                "source_break_start": case["source_break_start"],
                "source_break_end": case["source_break_end"],
                "description": case["description"],
                "ground_truth_label": case["ground_truth_label"],
                "model_answer": parsed.verdict,
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
            "parsed_verdict": row["model_answer"],
        })
        upsert_checkpoint(paths["checkpoint"], case_id, row)
        checkpoint[case_id] = row
        print(f"[{parse_status}] {case_id}: gt={case['ground_truth_label']} pred={row['model_answer']}")

    summary = _write_outputs(paths)
    _print_summary(summary, paths["summary"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
