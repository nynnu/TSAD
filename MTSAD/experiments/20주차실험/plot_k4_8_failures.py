"""
k=4-8 구간에서 recall@8<1.0인 실패 사례를 line plot으로 시각화하고
diagnosis 문서로 정리 -- plot_k1_3_failures.py와 동일한 방식, 대상 버킷과
저장 위치만 다름 (원인분석2/).
"""

import json
from pathlib import Path

from plot_k1_3_failures import CKPT_PATH, plot_case

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "원인분석2"
DIAG_DIR = OUT_DIR / "diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def run():
    ckpt = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
    fails = [v for v in ckpt.values() if v["bucket"] == "4-8" and v["recall_at_8"] < 1.0]
    fails.sort(key=lambda r: r["recall_at_8"])
    print(f"{len(fails)}개 실패 사례 플롯 생성 중...")

    test_cache = {}
    md_lines = [
        "# k=4-8 구간 실패 사례 (recall@8 < 1.0)",
        "",
        f"k=4-8 구간 총 16개 세그먼트 중 {len(fails)}개가 recall@8<1.0 (GT 채널을 top-8 안에서 "
        f"다 못 찾음). 굵은 검은선=GT 채널, 가는 색선=우리가 대신 잘못 고른 top-8 채널, "
        f"빨간 음영=실제 라벨링된 이상 구간.",
        "",
    ]

    for f in fails:
        entity, cs, ce = f["entity"], f["start"], f["end"]
        gt_channels, top8 = f["gt_channels"], f["top8"]
        img_path = plot_case(entity, cs, ce, gt_channels, top8, test_cache, out_dir=DIAG_DIR)
        rel = img_path.relative_to(OUT_DIR)
        print(f"  saved {rel}")

        md_lines += [
            f"## {entity} [{cs}-{ce}] (len={ce-cs+1})",
            "",
            f"- k={f['k']}, GT 채널={gt_channels}, 우리 top-8={top8}",
            f"- hit={f['hit']}/8, precision@8={f['precision_at_8']:.2f}, recall@8={f['recall_at_8']:.2f}",
            "",
            f"![{entity}_{cs}_{ce}]({rel})",
            "",
        ]

    out_md = OUT_DIR / "diagnosis.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved: {out_md}")
    print(f"Images: {DIAG_DIR}")


if __name__ == "__main__":
    run()
