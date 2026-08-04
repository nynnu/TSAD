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
from prompts import SANITY9_PROMPT
from sanity9_data_gen import generate_sample
from sanity9_parser import parse_causal_response
from visualize import render_multichannel_overlay, render_subplot_multichannel
from vlm_client import VLMClient, VLMResponse


# Sanity-9 tests root-cause / propagation reasoning across 6 channels with no
# fixed pairing: every sample randomly assigns one channel as "root" (the
# origin of a C1-C5 style break), 0-5 of the remaining channels as reactors
# (cascade = strong+abrupt, mild = weak+gradual, each independently
# same-type/homogeneous or different-type/heterogeneous vs the root, after a
# lag), and the rest as unrelated. This targets two failure modes Sanity-6
# already surfaced (misattribution, order/attention effects on later-listed
# pairs) plus root-cause/lag/heterogeneous-propagation reasoning that no prior
# sanity experiment tested. Two visualizations (stacked subplot vs overlay) are
# compared on the *same* underlying samples.
SANITY9_ROOT = ROOT / "results" / "sanity9" / "runs"

N_AFFECTED_VALUES = [0, 1, 2, 3, 4, 5]
LAG_VALUES = [10, 50]
VIS_CONDITIONS = ["overlay", "subplot"]
N_PER_SCENARIO = 30

SAMPLE_COLUMNS = [
    "case_id", "scenario", "n_affected", "lag", "vis_condition",
    "true_root", "pred_root", "root_correct",
    "true_onset", "pred_onset", "onset_abs_error",
    "model_confidence", "model_reason", "parse_status",
]

CHANNEL_COLUMNS = [
    "case_id", "scenario", "n_affected", "lag", "vis_condition", "channel",
    "true_role", "sub_role", "break_type", "homogeneous", "pred_role", "correct", "parse_status",
]


def _scenarios() -> list[tuple[int | None, int | None, str]]:
    out = [(n, lag, f"NA{n}_LAG{lag}") for n in N_AFFECTED_VALUES for lag in LAG_VALUES]
    out.append((None, None, "NORMAL"))
    return out


def _seed(n_affected: int | None, lag: int | None, index: int) -> int:
    if n_affected is None:
        return 900_000 + index
    return n_affected * 10_000 + lag * 100 + index


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "images": run_dir / "images",
        "logs": run_dir / "logs",
        "raw": run_dir / "raw",
        "diagnosis": run_dir / "diagnosis",
        "checkpoint": run_dir / "checkpoint.json",
        "infer_log": run_dir / "logs" / "sanity9_inferences.jsonl",
        "parse_failures": run_dir / "raw" / "parse_failures.jsonl",
        "samples_csv": run_dir / "raw" / "sanity9_samples.csv",
        "channels_csv": run_dir / "raw" / "sanity9_channels.csv",
        "summary": run_dir / "sanity9_summary.json",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mock_call(meta: dict) -> VLMResponse:
    raw = json.dumps({
        "root_cause_channel": meta["root_cause_channel"],
        "affected_channels": meta["affected_channels"],
        "unaffected_channels": meta["unaffected_channels"],
        "onset_time": meta["onset_time"],
        "reason": "Mock causal-cascade response for pipeline validation.",
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
            "vis_condition": e["vis_condition"],
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
                "vis_condition": e["vis_condition"],
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
    affected_only = channels[channels["true_role"] == "affected"]
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_per_scenario": N_PER_SCENARIO,
        "T": T,
        "n_affected_values": N_AFFECTED_VALUES,
        "lag_values": LAG_VALUES,
        "vis_conditions": VIS_CONDITIONS,
        "parse_status_counts": {str(k): int(v) for k, v in samples["parse_status"].value_counts(dropna=False).to_dict().items()},
        "overall_sample": _sample_metrics(samples),
        "overall_channel": _channel_metrics(channels),
        "by_vis_condition_sample": {str(k): _sample_metrics(v) for k, v in samples.groupby("vis_condition")},
        "by_vis_condition_channel": {str(k): _channel_metrics(v) for k, v in channels.groupby("vis_condition")},
        "by_scenario_sample": {str(k): _sample_metrics(v) for k, v in samples.groupby("scenario")},
        "by_n_affected_channel": {str(k): _channel_metrics(v) for k, v in channels.groupby("n_affected")},
        "by_lag_channel": {str(k): _channel_metrics(v) for k, v in channels.groupby("lag")},
        "by_homogeneous_channel": {str(k): _channel_metrics(v) for k, v in affected_only.groupby("homogeneous")},
    }


def _copy_example(paths: dict[str, Path], case_id: str, prefix: str) -> None:
    src = paths["images"] / f"{case_id}.png"
    if src.exists():
        shutil.copyfile(src, paths["diagnosis"] / f"{prefix}_{case_id}.png")


def _diagnosis(samples: pd.DataFrame, paths: dict[str, Path], summary: dict) -> None:
    paths["diagnosis"].mkdir(parents=True, exist_ok=True)
    scored = samples[samples["parse_status"] == "OK"].copy()
    for vis, sub in scored.groupby("vis_condition"):
        best = sub[sub["root_correct"] == True]  # noqa: E712
        worst = sub[sub["root_correct"] == False]  # noqa: E712
        if not best.empty:
            _copy_example(paths, best.sort_values("model_confidence", ascending=False).iloc[0]["case_id"], f"success_{vis}")
        if not worst.empty:
            _copy_example(paths, worst.sort_values("model_confidence", ascending=False).iloc[0]["case_id"], f"failure_{vis}")

    lines = ["# Sanity-9 Diagnosis (Causal Root-Cause / Propagation)", ""]
    lines.append(f"Overall (sample-level): {summary['overall_sample']}\n")
    lines.append(f"Overall (channel-level, affected-set precision/recall/f1): {summary['overall_channel']}\n")
    lines.append("## By visualization condition\n")
    for vis in VIS_CONDITIONS:
        lines.append(f"{vis} sample: {summary['by_vis_condition_sample'].get(vis)}\n")
        lines.append(f"{vis} channel: {summary['by_vis_condition_channel'].get(vis)}\n")
    lines.append("## By scenario (sample-level)\n")
    for scenario, metric in summary["by_scenario_sample"].items():
        lines.append(f"{scenario}: {metric}\n")
    lines.append("## By n_affected (channel-level)\n")
    for k, metric in summary["by_n_affected_channel"].items():
        lines.append(f"n_affected={k}: {metric}\n")
    lines.append("## By lag (channel-level)\n")
    for k, metric in summary["by_lag_channel"].items():
        lines.append(f"lag={k}: {metric}\n")
    lines.append("## Homogeneous vs heterogeneous propagation (channel-level, affected channels only)\n")
    for k, metric in summary["by_homogeneous_channel"].items():
        lines.append(f"homogeneous={k}: {metric}\n")
    (paths["diagnosis"] / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _run_dir(args) -> tuple[str, Path]:
    if args.resume:
        run_dir = SANITY9_ROOT / args.resume
        if not run_dir.exists() or not (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"Cannot resume missing Sanity-9 run: {args.resume}")
        return args.resume, run_dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return run_id, SANITY9_ROOT / run_id


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
    _diagnosis(samples, paths, summary)
    return summary


def _print_summary(summary: dict, summary_path: Path) -> None:
    print("\nSanity-9 Results")
    print(f"Parse status: {summary['parse_status_counts']}")
    print(f"Overall (sample): {summary['overall_sample']}")
    print(f"Overall (channel): {summary['overall_channel']}")
    for vis in VIS_CONDITIONS:
        print(f"  {vis}: sample={summary['by_vis_condition_sample'].get(vis)}")
        print(f"  {vis}: channel={summary['by_vis_condition_channel'].get(vis)}")
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

    scenarios = _scenarios()
    total_calls = len(scenarios) * N_PER_SCENARIO * len(VIS_CONDITIONS)
    print(f"Sanity-9 Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(
        f"Planned calls: {total_calls}; rough token budget: "
        f"~{int(total_calls * ESTIMATED_TOKENS_PER_CALL * 1.5):,} tokens "
        f"(6-channel image + longer JSON output than prior experiments)"
    )
    if not args.dry_run and not args.yes:
        ans = input("Proceed with real OpenAI API calls? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted.")

    checkpoint = load_checkpoint(paths["checkpoint"])
    client = None if args.dry_run else VLMClient(MODEL_NAME)

    for n_affected, lag, label in scenarios:
        for index in range(N_PER_SCENARIO):
            seed = _seed(n_affected, lag, index)
            channels, meta = generate_sample(
                n_affected=n_affected or 0, lag=lag or 0, seed=seed, t=T,
                force_normal=(n_affected is None),
            )

            for vis in VIS_CONDITIONS:
                case_id = f"{label}_{index:03d}_{vis}"
                image_path = paths["images"] / f"{case_id}.png"
                if vis == "overlay":
                    img_b64 = render_multichannel_overlay(channels, image_path)
                else:
                    img_b64 = render_subplot_multichannel(channels, image_path)

                if should_skip(checkpoint.get(case_id)):
                    print(f"[SKIP] {case_id}")
                    continue

                if args.dry_run:
                    resp = _mock_call(meta)
                else:
                    resp = client.call(SANITY9_PROMPT, img_b64)

                base_entry = {
                    "scenario": label,
                    "n_affected": n_affected,
                    "lag": lag,
                    "vis_condition": vis,
                    "true_root": meta["root_cause_channel"],
                    "true_onset": meta["onset_time"],
                    "roles": meta["roles"],
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
                print(f"[{entry['status']}] {case_id}: true_root={meta['root_cause_channel']} pred_root={entry['pred_root']}")

    summary = _write_outputs(paths)
    _print_summary(summary, paths["summary"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
