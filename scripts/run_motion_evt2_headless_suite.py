#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EVT2 motion reference diagnostics followed by real policy inference"
    )
    parser.add_argument("--config", default="configs/motion_evt2.json")
    parser.add_argument("--output-dir", default="artifacts/motion_evt2_headless_suite")
    parser.add_argument("--debug-interval", type=int, default=100)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = (project / args.config).resolve()
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    missing_summary = False

    # Reference playback is diagnostic only and may fall. Always run it before policy inference.
    for mode in ("reference", "policy"):
        for motion in ("a", "b"):
            key = f"{motion}_{mode}"
            summary_path = output_dir / f"{key}_summary.json"
            summary_path.unlink(missing_ok=True)
            command = [
                sys.executable,
                str(project / "run_motion_evt2.py"),
                "--config",
                str(config),
                "--motion",
                motion,
                "--mode",
                mode,
                "--headless",
                "--no-realtime",
                "--debug-interval",
                str(args.debug_interval),
                "--log",
                str(output_dir / f"{key}.csv"),
                "--summary",
                str(summary_path),
            ]
            print(f"[SUITE] running motion={motion} mode={mode}", flush=True)
            completed = subprocess.run(command, cwd=project, check=False)
            if summary_path.is_file():
                result = json.loads(summary_path.read_text(encoding="utf-8"))
                result["process_returncode"] = completed.returncode
                results[key] = result
            else:
                missing_summary = True
                results[key] = {
                    "passed": False,
                    "error": "summary not produced",
                    "process_returncode": completed.returncode,
                }

    policy_passed = all(results[f"{motion}_policy"].get("policy_success", False) for motion in ("a", "b"))
    diagnostics_complete = not missing_summary and all(
        f"{motion}_{mode}" in results for mode in ("reference", "policy") for motion in ("a", "b")
    )
    suite_summary = {
        "passed": policy_passed and diagnostics_complete,
        "policy_suite_passed": policy_passed,
        "reference_results_are_diagnostic_only": True,
        "diagnostics_complete": diagnostics_complete,
        "runs": results,
    }
    suite_path = output_dir / "suite_summary.json"
    suite_path.write_text(json.dumps(suite_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[SUITE] {'PASS' if suite_summary['passed'] else 'FAIL'}: {suite_path}")
    if not suite_summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
