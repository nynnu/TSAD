import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from checkpoint import load_checkpoint, save_checkpoint, should_skip
from config import ESTIMATED_TOKENS_PER_CALL, MODEL_NAME, ROOT, TEMPERATURE
from prompts import build_oracle9_prompt
from sanity9_parser import parse_causal_response
from vlm_client import VLMClient, VLMResponse


# Oracle-Clean: reuses the exact 390 unique ground-truth samples behind the
# Sanity-9 run (one row per underlying sample, vis_condition == "overlay" is
# picked arbitrarily since both vis conditions share identical `roles` for a
# given case) but replaces the image with a text-only description of the same
# ground truth. Tests the LLM's causal-reasoning ceiling with the Stage-1
# vision bottleneck removed entirely.
SANITY9_SOURCE = ROOT / "results" / "sanity9" / "runs" / "20260716_112728" / "checkpoint.json"
ORACLE9_ROOT = ROOT / "results" / "oracle9" / "runs"

SAMPLE_COLUMNS = [
    "case_id", "scenario", "n_affected", "lag",
    "true_root", "pred_root", "root_correct",
    "true_onset", "pred_onset", "onset_abs_error",
    "model_confidence", "model_reason", "parse_status",
]

CHANNEL_COLUMNS = [
    "case_id", "scenario", "n_affected", "lag", "channel",
    "true_role", "sub_role", "break_type", "homogeneous", "pred_role", "correct", "parse_status",
]


def _load_source_cases() -> dict[str, dict]:
    if not SANITY9_SOURCE.exists():
        raise SystemExit(f"Sanity-9 source checkpoint not found: {SANITY9_SOURCE}")
    all_cases = load_checkpoint(SANITY9_SOURCE)
    overlay_only = {k: v for k, v in all_cases.items() if v["vis_condition"] == "overlay"}
    return {k.removesuffix("_overlay"): v for k, v in overlay_only.items()}


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "logs": run_dir / "logs",
        "raw": run_dir / "raw",
        "checkpoint": run_dir / "checkpoint.json",
        "infer_log": run_dir / "logs" / "oracle9_inferences.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
        "samples_csv": run_dir / "raw" / "oracle9_samples.csv",
        "channels_csv": run_dir / "raw" / "oracle9_channels.csv",
        "summary": run_dir / "oracle9_summary.json",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mock_call(source_entry: dict) -> VLMResponse:
    raw = json.dumps({
        "root_cause_channel": source_entry["true_root"],
        "affected_channels": [n for n, info in source_entry["roles"].items() if info["role"] in ("cascade", "mild")],
        "unaffected_channels": [n for n, info in source_entry["roles"].items() if info["role"] == "unrelated"],
        "onset_time": source_entry["true_onset"],
        "reason": "Mock oracle response for pipeline validation.",
        "confidence": 0.99,
    })
    return VLMResponse("OK", raw_response=raw, input_tokens=0, output_tokens=0, total_tokens=0, latency_seconds=0.0)


def _expand_sample_rows(checkpoint: dict) -> list[dict]:
    rows = []
    for case_id, e in checkpoint.items():
        true_onset, pred_onset = e.get("true_onset"), e.get("pred_onset")
        onset_err = abs(pred_onset - true_onset) if (true_onset is not None and pred_onset is not None) else None
        pred_root = e.get("pred_root")
        rows.append({
            "case_id": case_id,
            "scenario": e["scenario"],
            "n_affected": e["n_affected"],
            "lag": e["lag"],
            "true_root": e["true_root"],
            "pred_root": pred_root,
            "root_correct": (pred_root == e["true_root"]) if pred_root is not None else None,
            "true_onset": true_onset,
            "pred_onset": pred_onset,
            "onset_abs_error": onset_err,
            "model_confidence": e.get("model_confidence"),
            "model_reason": e.get("model_reason"),
            "parse_status": e.get("status"),
        })
    return rows


def _expand_channel_rows(checkpoint: dict) -> list[dict]:
    rows = []
    for case_id, e in checkpoint.items():
        roles = e.get("roles") or {}
        pred_affected = set(e.get("pred_affected") or [])
        pred_unaffected = set(e.get("pred_unaffected") or [])
        pred_root = e.get("pred_root")
        for name, info in roles.items():
            if info["role"] == "root":
                continue
            true_role = "affected" if info["role"] in ("cascade", "mild") else "unrelated"
            if name in pred_affected:
                pred_role = "affected"
            elif name in pred_unaffected:
                pred_role = "unrelated"
            elif name == pred_root:
                pred_role = "root"
            else:
                pred_role = "unlabeled"
            rows.append({
                "case_id": case_id,
                "scenario": e["scenario"],
                "n_affected": e["n_affected"],
                "lag": e["lag"],
                "channel": name,
                "true_role": true_role,
                "sub_role": info["role"],
                "break_type": info.get("break_type"),
                "homogeneous": info.get("homogeneous"),
                "pred_role": pred_role,
                "correct": pred_role == true_role,
                "parse_status": e.get("status"),
            })
    return rows


def _sample_metrics(df: pd.DataFrame) -> dict:
    scored = df[df["parse_status"] == "OK"]
    n = len(scored)
    if n == 0:
        return {"n": 0, "root_accuracy": None, "onset_mae": None, "n_onset_valid": 0}
    onset_valid = scored[scored["onset_abs_error"].notna()]
    return {
        "n": int(n),
        "root_accuracy": float((scored["root_correct"] == True).mean()),  # noqa: E712
        "onset_mae": float(onset_valid["onset_abs_error"].mean()) if len(onset_valid) else None,
        "n_onset_valid": int(len(onset_valid)),
    }


def _channel_metrics(df: pd.DataFrame) -> dict:
    scored = df[df["parse_status"] == "OK"]
    n = len(scored)
    if n == 0:
        return {"n": 0, "precision": None, "recall": None, "f1": None}
    tp = int(((scored.true_role == "affected") & (scored.pred_role == "affected")).sum())
    fp = int(((scored.true_role == "unrelated") & (scored.pred_role == "affected")).sum())
    fn = int(((scored.true_role == "affected") & (scored.pred_role != "affected")).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"n": int(n), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _build_summary(samples: pd.DataFrame, channels: pd.DataFrame) -> dict:
    normal = samples[samples["scenario"] == "NORMAL"]
    normal_scored = normal[normal["parse_status"] == "OK"]
    normal_fp_rate = (
        float((normal_scored["pred_root"] != "none").mean()) if len(normal_scored) else None
    )
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "source_checkpoint": str(SANITY9_SOURCE),
        "parse_status_counts": {str(k): int(v) for k, v in samples["parse_status"].value_counts(dropna=False).to_dict().items()},
        "overall_sample": _sample_metrics(samples),
        "overall_channel": _channel_metrics(channels),
        "normal_fp_rate": normal_fp_rate,
        "n_normal_scored": int(len(normal_scored)),
        "by_scenario_sample": {str(k): _sample_metrics(v) for k, v in samples.groupby("scenario")},
        "by_n_affected_sample": {str(k): _sample_metrics(v) for k, v in samples[samples["scenario"] != "NORMAL"].groupby("n_affected")},
        "by_n_affected_channel": {str(k): _channel_metrics(v) for k, v in channels.groupby("n_affected")},
        "by_lag_channel": {str(k): _channel_metrics(v) for k, v in channels.groupby("lag")},
        "by_homogeneous_channel": {
            str(k): _channel_metrics(v)
            for k, v in channels[channels["true_role"] == "affected"].groupby("homogeneous")
        },
    }


def _write_outputs(paths: dict[str, Path]) -> dict:
    checkpoint = load_checkpoint(paths["checkpoint"])
    samples = pd.DataFrame(_expand_sample_rows(checkpoint))
    channels = pd.DataFrame(_expand_channel_rows(checkpoint))
    if samples.empty:
        raise SystemExit("No checkpoint rows produced.")
    paths["raw"].mkdir(parents=True, exist_ok=True)
    samples[SAMPLE_COLUMNS].to_csv(paths["samples_csv"], index=False)
    channels[CHANNEL_COLUMNS].to_csv(paths["channels_csv"], index=False)
    summary = _build_summary(samples, channels)
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_dir = ORACLE9_ROOT / args.resume
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume missing Oracle-9 run: {args.resume}")
        return args.resume, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, ORACLE9_ROOT / run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use mocked VLM responses; no API cost.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation for real API run.")
    parser.add_argument("--resume", default=None, help="Resume an existing run_id.")
    args = parser.parse_args()

    run_id, run_dir = _run_dir(args)
    paths = _paths(run_dir)
    for key in ["logs", "raw"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    source_cases = _load_source_cases()
    total_calls = len(source_cases)
    print(f"Oracle-9 Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Source: {SANITY9_SOURCE} ({total_calls} unique ground-truth samples)")
    print(f"Planned calls: {total_calls}; rough token budget: ~{int(total_calls * ESTIMATED_TOKENS_PER_CALL * 0.5):,} tokens (text-only, no image)")
    if not args.dry_run and not args.yes:
        ans = input("Proceed with real OpenAI API calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = None if args.dry_run else VLMClient(MODEL_NAME)

    for case_id, source_entry in source_cases.items():
        if should_skip(checkpoint.get(case_id)):
            print(f"[SKIP] {case_id}")
            continue

        prompt = build_oracle9_prompt(source_entry["roles"])
        if args.dry_run:
            resp = _mock_call(source_entry)
        else:
            resp = client.call_text(prompt)

        base_entry = {
            "scenario": source_entry["scenario"],
            "n_affected": source_entry["n_affected"],
            "lag": source_entry["lag"],
            "true_root": source_entry["true_root"],
            "true_onset": source_entry["true_onset"],
            "roles": source_entry["roles"],
        }

        if resp.status == "API_ERROR":
            entry = {**base_entry, "status": "API_ERROR", "pred_root": None, "pred_affected": [],
                     "pred_unaffected": [], "pred_onset": None, "model_confidence": None,
                     "model_reason": resp.error or ""}
        else:
            parsed = parse_causal_response(resp.raw_response)
            if parsed.status == "PARSE_ERROR":
                _append_jsonl(paths["parse_failures"], {
                    "case_id": case_id,
                    "failure_reason": parsed.failure_reason,
                    "raw_response": resp.raw_response,
                })
                entry = {**base_entry, "status": "PARSE_ERROR", "pred_root": None, "pred_affected": [],
                         "pred_unaffected": [], "pred_onset": None, "model_confidence": None,
                         "model_reason": parsed.failure_reason or ""}
            else:
                entry = {
                    **base_entry, "status": "OK",
                    "pred_root": parsed.root_cause_channel,
                    "pred_affected": parsed.affected_channels,
                    "pred_unaffected": parsed.unaffected_channels,
                    "pred_onset": parsed.onset_time,
                    "model_confidence": parsed.confidence,
                    "model_reason": parsed.reason,
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
            "parse_status": entry["status"],
            "pred_root": entry["pred_root"],
        })
        checkpoint[case_id] = entry
        save_checkpoint(paths["checkpoint"], checkpoint)
        print(f"[{entry['status']}] {case_id}: true_root={source_entry['true_root']} pred_root={entry['pred_root']}")

    summary = _write_outputs(paths)
    print("\nOracle-9 Results")
    print(f"Parse status: {summary['parse_status_counts']}")
    print(f"Overall (sample): {summary['overall_sample']}")
    print(f"Overall (channel): {summary['overall_channel']}")
    print(f"NORMAL FP rate: {summary['normal_fp_rate']} (n={summary['n_normal_scored']})")
    print(f"\nSaved summary: {paths['summary']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
