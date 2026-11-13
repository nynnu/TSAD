import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from checkpoint import load_checkpoint, save_checkpoint, should_skip
from config import ESTIMATED_TOKENS_PER_CALL, MODEL_NAME, ROOT, T, TEMPERATURE
from multichannel_data_gen import generate_scene
from prompts import build_sanity6_prompt
from sanity6_parser import parse_multichannel_response
from visualize import render_multichannel_overlay
from vlm_client import VLMClient, VLMResponse


SANITY6_ROOT = ROOT / "results" / "sanity6" / "runs"

# (n_channels, n_broken_pairs, scenario_label). n_broken_pairs=0 is the negative
# control at each scale; "single" tests minimal-signal detection amid clutter;
# "multi" tests whether simultaneous breaks all get caught or attention narrows
# to just one (per the design doc's attention-bias hypothesis).
SCENARIOS: list[tuple[int, int, str]] = [
    (2, 0, "N2_none"), (2, 1, "N2_single"),
    (4, 0, "N4_none"), (4, 1, "N4_single"), (4, 2, "N4_multi"),
    (8, 0, "N8_none"), (8, 1, "N8_single"), (8, 2, "N8_multi"),
]
N_PER_SCENARIO = 15

RAW_COLUMNS = [
    "case_id", "n_channels", "break_scenario", "pair_name", "pair_position",
    "break_type", "ground_truth", "model_answer", "correct",
    "model_confidence", "model_reason", "parse_status",
]


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "images": run_dir / "images",
        "logs": run_dir / "logs",
        "raw": run_dir / "raw",
        "diagnosis": run_dir / "diagnosis",
        "checkpoint": run_dir / "checkpoint.json",
        "infer_log": run_dir / "logs" / "sanity6_inferences.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
        "raw_csv": run_dir / "raw" / "sanity6_raw.csv",
        "summary": run_dir / "sanity6_summary.json",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mock_call(ground_truth: dict[str, str]) -> VLMResponse:
    raw = json.dumps({
        "pairs": ground_truth,
        "reason": "Mock multi-channel response for pipeline validation.",
        "confidence": 0.99,
    })
    return VLMResponse("OK", raw_response=raw, input_tokens=0, output_tokens=0, total_tokens=0, latency_seconds=0.0)


def _expand_rows(checkpoint: dict) -> list[dict]:
    rows = []
    for case_id, entry in checkpoint.items():
        pair_names = entry["pair_names"]
        gt = entry["ground_truth"]
        break_types = entry.get("break_types", {})
        model_answer = entry.get("model_answer") or {}
        for pos, pair_name in enumerate(pair_names):
            ans = model_answer.get(pair_name)
            rows.append({
                "case_id": case_id,
                "n_channels": entry["n_channels"],
                "break_scenario": entry["break_scenario"],
                "pair_name": pair_name,
                "pair_position": pos,
                "break_type": break_types.get(pair_name),
                "ground_truth": gt[pair_name],
                "model_answer": ans,
                "correct": (ans == gt[pair_name]) if ans is not None else None,
                "model_confidence": entry.get("model_confidence"),
                "model_reason": entry.get("model_reason"),
                "parse_status": entry.get("status"),
            })
    return rows


def _pair_metrics(df: pd.DataFrame) -> dict:
    scored = df[(df["parse_status"] == "OK") & df["model_answer"].notna()]
    n = len(scored)
    if n == 0:
        return {
            "n": 0, "accuracy": None, "precision_broken": None, "recall_broken": None,
            "f1_broken": None, "false_positive_rate": None,
        }
    y_true, y_pred = scored["ground_truth"], scored["model_answer"]
    tp = int(((y_true == "broken") & (y_pred == "broken")).sum())
    fp = int(((y_true == "maintained") & (y_pred == "broken")).sum())
    fn = int(((y_true == "broken") & (y_pred == "maintained")).sum())
    tn = int(((y_true == "maintained") & (y_pred == "maintained")).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n": n,
        "accuracy": float((tp + tn) / n),
        "precision_broken": float(precision),
        "recall_broken": float(recall),
        "f1_broken": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
    }


def _build_summary(df: pd.DataFrame) -> dict:
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_per_scenario": N_PER_SCENARIO,
        "T": T,
        "scenarios": SCENARIOS,
        "parse_status_counts": {str(k): int(v) for k, v in df["parse_status"].value_counts(dropna=False).to_dict().items()},
        "overall": _pair_metrics(df),
        "by_n_channels": {str(k): _pair_metrics(v) for k, v in df.groupby("n_channels")},
        "by_scenario": {str(k): _pair_metrics(v) for k, v in df.groupby("break_scenario")},
        "by_pair_position": {str(k): _pair_metrics(v) for k, v in df.groupby("pair_position")},
    }


def _copy_example(paths: dict[str, Path], case_id: str, prefix: str) -> None:
    src = paths["images"] / f"{case_id}.png"
    if src.exists():
        shutil.copyfile(src, paths["diagnosis"] / f"{prefix}_{case_id}.png")


def _diagnosis(df: pd.DataFrame, paths: dict[str, Path], summary: dict) -> None:
    paths["diagnosis"].mkdir(parents=True, exist_ok=True)
    scored = df[df["parse_status"] == "OK"].copy()
    if not scored.empty:
        scene_acc = scored.groupby(["case_id", "n_channels"])["correct"].mean().reset_index()
        for n_channels, sub in scene_acc.groupby("n_channels"):
            best = sub.sort_values("correct", ascending=False).iloc[0]
            worst = sub.sort_values("correct", ascending=True).iloc[0]
            _copy_example(paths, best["case_id"], f"best_N{n_channels}")
            _copy_example(paths, worst["case_id"], f"worst_N{n_channels}")

    lines = ["# Sanity-6 Diagnosis (Multi-Channel Scaling)", ""]
    lines.append(
        "Sanity-6 tests how many overlaid channels a single image can hold before GPT-4o's "
        "per-pair maintained/broken judgment degrades, and whether simultaneous breaks across "
        "multiple pairs get missed due to attention narrowing to one salient pair.\n"
    )
    lines.append(f"Overall: {summary['overall']}\n")
    lines.append("## By channel count\n")
    for n_channels, metric in summary["by_n_channels"].items():
        lines.append(f"N={n_channels}: {metric}\n")
    lines.append("## By scenario\n")
    for scenario, metric in summary["by_scenario"].items():
        lines.append(f"{scenario}: {metric}\n")
    lines.append("## By pair position within the image (0 = first pair listed)\n")
    for pos, metric in summary["by_pair_position"].items():
        lines.append(f"position {pos}: {metric}\n")
    (paths["diagnosis"] / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_dir = SANITY6_ROOT / args.resume
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume missing Sanity-6 run: {args.resume}")
        return args.resume, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, SANITY6_ROOT / run_id


def _write_outputs(paths: dict[str, Path]) -> dict:
    checkpoint = load_checkpoint(paths["checkpoint"])
    rows = _expand_rows(checkpoint)
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
    print("\nSanity-6 Results")
    print(f"Parse status: {summary['parse_status_counts']}")
    print(f"Overall: {summary['overall']}")
    for n_channels, metric in summary["by_n_channels"].items():
        print(f"N={n_channels}: {metric}")
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

    total_calls = len(SCENARIOS) * N_PER_SCENARIO
    print(f"Sanity-6 Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Planned calls: {total_calls}; rough token budget: ~{total_calls * ESTIMATED_TOKENS_PER_CALL:,} tokens "
          f"(images have more channels/legend entries than prior sanity experiments, so real usage may run higher)")
    if not args.dry_run and not args.yes:
        ans = input("Proceed with real OpenAI API calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = None if args.dry_run else VLMClient(MODEL_NAME)

    for n_channels, n_broken_pairs, label in SCENARIOS:
        for index in range(N_PER_SCENARIO):
            case_id = f"{label}_{index:03d}"
            scene_seed = 10_000 * n_channels + 100 * n_broken_pairs + index
            scene = generate_scene(n_channels, n_broken_pairs, seed=scene_seed, t=T)

            image_path = paths["images"] / f"{case_id}.png"
            img_b64 = render_multichannel_overlay(scene["channels"], image_path)

            if should_skip(checkpoint.get(case_id)):
                print(f"[SKIP] {case_id}")
                continue

            prompt = build_sanity6_prompt(scene["pair_names"])
            if args.dry_run:
                resp = _mock_call(scene["ground_truth"])
            else:
                resp = client.call(prompt, img_b64)

            base_entry = {
                "n_channels": n_channels,
                "break_scenario": label,
                "pair_names": scene["pair_names"],
                "ground_truth": scene["ground_truth"],
                "break_types": scene["break_types"],
            }

            if resp.status == "API_ERROR":
                entry = {**base_entry, "status": "API_ERROR", "model_answer": {}, "model_confidence": None, "model_reason": resp.error or ""}
            else:
                parsed = parse_multichannel_response(resp.raw_response, scene["pair_names"])
                if parsed.status == "PARSE_ERROR":
                    _append_jsonl(paths["parse_failures"], {
                        "case_id": case_id,
                        "failure_reason": parsed.failure_reason,
                        "raw_response": resp.raw_response,
                    })
                    entry = {**base_entry, "status": "PARSE_ERROR", "model_answer": {}, "model_confidence": None, "model_reason": parsed.failure_reason or ""}
                else:
                    entry = {
                        **base_entry, "status": "OK",
                        "model_answer": parsed.pair_answers,
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
                "model_answer": entry["model_answer"],
            })
            checkpoint[case_id] = entry
            save_checkpoint(paths["checkpoint"], checkpoint)
            print(f"[{entry['status']}] {case_id}: gt={scene['ground_truth']} pred={entry['model_answer']}")

    summary = _write_outputs(paths)
    _print_summary(summary, paths["summary"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
