import argparse
import base64
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from checkpoint import checkpoint_to_rows, load_checkpoint, should_skip, upsert_checkpoint
from config import MODEL_NAME, RESULTS_ROOT, TEMPERATURE
from prompts import build_sanity2_judge_prompt
from sanity2_parser import parse_reason_judgment
from vlm_client import MockVLMClient, VLMClient, VLMResponse


SANITY2_ROOT = Path(__file__).resolve().parent / "results" / "sanity2" / "runs"


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "raw": run_dir / "raw",
        "logs": run_dir / "logs",
        "diagnosis": run_dir / "diagnosis",
        "checkpoint": run_dir / "checkpoint.json",
        "raw_csv": run_dir / "raw" / "sanity2_raw.csv",
        "summary": run_dir / "sanity2_summary.json",
        "judge_log": run_dir / "logs" / "sanity2_judgments.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _latest_complete_sanity1_run() -> str:
    candidates = sorted((RESULTS_ROOT).glob("*"), reverse=True)
    for run_dir in candidates:
        summary = run_dir / "sanity1_summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data.get("status_counts", {}).get("OK") == 210:
            return run_dir.name
    raise SystemExit("No complete Sanity-1 run found. Pass --source-run explicitly.")


def _mock_reason_type(reason: str) -> str:
    text = str(reason).lower()
    relational_words = ["channel a", "channel b", "together", "relationship", "phase", "sync", "aligned", "diverge"]
    if sum(word in text for word in relational_words) >= 2:
        return "relational"
    if any(word in text for word in ["flat", "amplitude", "noise", "spike", "channel"]):
        return "single_channel"
    return "vague"


def _mock_call(reason: str) -> VLMResponse:
    reason_type = _mock_reason_type(reason)
    raw = json.dumps({
        "reason_type": reason_type,
        "rationale": "Mock classification for pipeline validation.",
    })
    return VLMResponse("OK", raw_response=raw, input_tokens=0, output_tokens=0, total_tokens=0, latency_seconds=0.0)


def _summary(df: pd.DataFrame) -> dict:
    ok = df[df["judge_status"] == "OK"].copy()
    ok["answer_correct"] = ok["ground_truth_label"] == ok["model_answer"]
    reason_counts = ok["reason_type"].value_counts().to_dict()
    by_case = {}
    for case, sub in ok.groupby("case_type"):
        by_case[case] = {
            "n": int(len(sub)),
            "accuracy": float(sub["answer_correct"].mean()) if len(sub) else None,
            "reason_type_counts": {str(k): int(v) for k, v in sub["reason_type"].value_counts().to_dict().items()},
            "correct_relational_rate": float(((sub["answer_correct"]) & (sub["reason_type"] == "relational")).mean()) if len(sub) else None,
        }
    matrix = {}
    for correctness, sub in ok.groupby("answer_correct"):
        matrix["correct" if correctness else "incorrect"] = {
            str(k): int(v) for k, v in sub["reason_type"].value_counts().to_dict().items()
        }
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_total": int(len(df)),
        "judge_status_counts": {str(k): int(v) for k, v in df["judge_status"].value_counts().to_dict().items()},
        "reason_type_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "correctness_by_reason_type": matrix,
        "per_case": by_case,
    }


def _diagnosis(df: pd.DataFrame, paths: dict[str, Path], source_dir: Path) -> None:
    paths["diagnosis"].mkdir(parents=True, exist_ok=True)
    ok = df[df["judge_status"] == "OK"].copy()
    ok["answer_correct"] = ok["ground_truth_label"] == ok["model_answer"]

    lines = ["# Sanity-2 Diagnosis", ""]
    for case, sub in ok.groupby("case_type"):
        counts = sub["reason_type"].value_counts().to_dict()
        acc = float(sub["answer_correct"].mean())
        rel_correct = float(((sub["answer_correct"]) & (sub["reason_type"] == "relational")).mean())
        lines.append(
            f"## {case}\nAccuracy from Sanity-1 answers: {acc:.3f}. "
            f"Reason counts: {counts}. Correct+relational rate: {rel_correct:.3f}.\n"
        )
        examples = sub.sort_values(["answer_correct", "model_confidence"], ascending=[False, False]).head(1)
        for _, row in examples.iterrows():
            src = source_dir / "images" / f"{row['case_id']}.png"
            if src.exists():
                shutil.copyfile(src, paths["diagnosis"] / f"example_{case}_{row['reason_type']}_{row['case_id']}.png")
    (paths["diagnosis"] / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_dir = SANITY2_ROOT / args.resume
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume missing Sanity-2 run: {args.resume}")
        return args.resume, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, SANITY2_ROOT / run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", default=None, help="Sanity-1 run id to analyze.")
    ap.add_argument("--resume", default=None, help="Resume a Sanity-2 run id.")
    ap.add_argument("--dry-run", action="store_true", help="Mock judge; no API cost.")
    ap.add_argument("--yes", action="store_true", help="Skip real-run confirmation.")
    args = ap.parse_args()

    source_run = args.source_run or _latest_complete_sanity1_run()
    source_dir = RESULTS_ROOT / source_run
    raw_path = source_dir / "raw" / "sanity1_raw.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing Sanity-1 raw CSV: {raw_path}")

    run_id, run_dir = _run_dir(args)
    paths = _paths(run_dir)
    for key in ["raw", "logs", "diagnosis"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    print(f"Sanity-2 Run ID: {run_id}")
    print(f"Source Sanity-1 Run ID: {source_run}")
    df1 = pd.read_csv(raw_path)
    df1 = df1[df1["parse_status"] == "OK"].copy()
    print(f"Judgments planned: {len(df1)}")
    if not args.dry_run and not args.yes:
        ans = input("Proceed with GPT-4o judge calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = None if args.dry_run else VLMClient(MODEL_NAME)

    for _, row1 in df1.iterrows():
        case_id = str(row1["case_id"])
        if should_skip(checkpoint.get(case_id)):
            print(f"[SKIP] {case_id}")
            continue

        prompt = build_sanity2_judge_prompt(str(row1["model_answer"]), str(row1["model_reason"]))
        img_path = source_dir / "images" / f"{case_id}.png"
        if args.dry_run:
            resp = _mock_call(str(row1["model_reason"]))
        else:
            resp = client.call(prompt, _image_b64(img_path))

        if resp.status == "API_ERROR":
            judge_status = "API_ERROR"
            parsed_reason_type = None
            rationale = resp.error or ""
            failure_reason = resp.error or "api_error"
        else:
            parsed = parse_reason_judgment(resp.raw_response)
            judge_status = parsed.status
            parsed_reason_type = parsed.reason_type
            rationale = parsed.rationale or parsed.failure_reason or ""
            failure_reason = parsed.failure_reason
            if parsed.status == "PARSE_ERROR":
                _append_jsonl(paths["parse_failures"], {
                    "case_id": case_id,
                    "failure_reason": failure_reason,
                    "raw_response": resp.raw_response,
                })

        out = {
            "status": judge_status,
            "judge_status": judge_status,
            "case_type": row1["case_type"],
            "ground_truth_label": row1["ground_truth_label"],
            "model_answer": row1["model_answer"],
            "model_confidence": float(row1["model_confidence"]),
            "model_reason": row1["model_reason"],
            "reason_type": parsed_reason_type,
            "judge_rationale": rationale,
        }
        _append_jsonl(paths["judge_log"], {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "model_name": MODEL_NAME,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "total_tokens": resp.total_tokens,
            "latency_seconds": resp.latency_seconds,
            "raw_response": resp.raw_response,
            "judge_status": judge_status,
            "reason_type": parsed_reason_type,
        })
        upsert_checkpoint(paths["checkpoint"], case_id, out)
        checkpoint[case_id] = out
        print(f"[{judge_status}] {case_id}: {parsed_reason_type}")

    rows = checkpoint_to_rows(load_checkpoint(paths["checkpoint"]))
    df = pd.DataFrame(rows)
    df.to_csv(paths["raw_csv"], index=False)
    summary = _summary(df)
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _diagnosis(df, paths, source_dir)

    print("\nSanity-2 Results")
    print(f"Judge status: {summary['judge_status_counts']}")
    print(f"Reason counts: {summary['reason_type_counts']}")
    print(f"Correctness x reason type: {summary['correctness_by_reason_type']}")
    for case, info in summary["per_case"].items():
        print(f"{case}: acc={info['accuracy']:.3f}, correct+relational={info['correct_relational_rate']:.3f}, counts={info['reason_type_counts']}")
    print(f"\nSaved summary: {paths['summary']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
