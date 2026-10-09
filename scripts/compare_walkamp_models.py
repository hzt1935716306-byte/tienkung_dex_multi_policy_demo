#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.model_audit import compare_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MuJoCo dynamics relevant to WALKAMP")
    parser.add_argument("--reference", default="assets/official_evt2/urdf/evt2.xml")
    parser.add_argument("--candidate", default="assets/mjcf/dex_evt_full.xml")
    parser.add_argument("--output", default="")
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--allow-differences", action="store_true")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    report = compare_models(
        mujoco.MjModel.from_xml_path(str(reference_path)),
        mujoco.MjModel.from_xml_path(str(candidate_path)),
        args.tolerance,
    )
    report["reference"] = str(reference_path)
    report["candidate"] = str(candidate_path)
    output = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"[INFO] report: {output_path}")
    print(f"[RESULT] equivalent={report['equivalent']} mismatches={report['mismatch_counts']}")
    if not report["equivalent"] and not args.allow_differences:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
