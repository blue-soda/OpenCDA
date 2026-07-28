# -*- coding: utf-8 -*-
"""Run dense-LiDAR Table 2 fusion/global-box diagnostics."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table2_dense_20260729"
DATASET_ROOT = Path(r"D:\Data\Carla")
SCENARIO_ID = "2026_07_29_02_32_08"


BASE_ARGS = [
    "-m", "opencda.tools.offline_inference",
    "--dataset-root", str(DATASET_ROOT),
    "--scenario-id", SCENARIO_ID,
    "--ego-cav-id", "1",
    "--max-frames", "41",
    "--fusion-method", "early",
    "--num-channels", "10",
    "--bandwidth-mhz", "40",
    "--channel-estimator", "ns3",
    "--ns3-tb-size-bytes", "899",
    "--ns3-slot-duration-ms", "0.5",
    "--ns3-subchannel-prbs", "10",
    "--ns3-symbols-per-slot", "12",
    "--ns3-mcs", "28",
    "--communication-deadline-ms", "60",
]


RUNS = [
    {
        "name": "clustered_early_only",
        "args": [
            "--sgcp-constrained",
            "--clustering", "potential_verified_cov_coalition_game",
            "--resource-allocation", "cov_potential_game",
            "--sgcp-receiver-policy", "all-cluster-heads",
            "--sgcp-upload-mode", "grid",
            "--n-max", "5",
            "--rho-th", "1",
            "--sgcp-frame-mbps-budget", "60",
            "--max-senders-per-receiver", "2",
        ],
    },
    {
        "name": "pcs_global_box_k2",
        "args": [
            "--sgcp-constrained",
            "--clustering", "singleton",
            "--resource-allocation", "fullperception_pcs",
            "--sgcp-receiver-policy", "all-cavs",
            "--sgcp-upload-mode", "grid",
            "--sgcp-inter-cluster-late-fusion",
            "--pcs-frame-deadline-ms", "60",
            "--pcs-blind-spot-min-division", "4",
            "--pcs-blind-spot-radius", "4",
            "--pcs-min-spot-grids", "128",
            "--pcs-communication-range-m", "35",
            "--max-senders-per-receiver", "2",
        ],
    },
    {
        "name": "edgecooper_global_box_k2",
        "args": [
            "--selective-sharing-baseline", "edgecooper_global",
            "--clustering", "singleton",
            "--sgcp-receiver-policy", "all-cavs",
            "--sgcp-upload-mode", "grid",
            "--sgcp-inter-cluster-late-fusion",
            "--selective-member-budget", "3",
            "--selective-grid-budget", "117",
            "--selective-frame-deadline-ms", "60",
            "--edgecooper-global-comm-range-m", "35",
            "--max-senders-per-receiver", "2",
        ],
    },
]


def run_one(name, extra_args, force=False):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT / ("%s.log" % name)
    trace_path = ARTIFACT / ("%s_trace.csv" % name)
    eval_path = ARTIFACT / ("%s_eval_stats.csv" % name)
    if log_path.exists() and trace_path.exists() and not force:
        print("skip existing %s" % name)
        return
    cmd = [
        sys.executable,
        *BASE_ARGS,
        *extra_args,
        "--sgcp-trace-output", str(trace_path),
        "--eval-stats-output", str(eval_path),
    ]
    print("running %s" % name)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: %s\n\n" % " ".join(cmd))
        log.flush()
        subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("done %s" % name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    only = None
    if args.only:
        only = {item.strip() for item in args.only.split(",") if item.strip()}
    for run in RUNS:
        if only is not None and run["name"] not in only:
            continue
        run_one(run["name"], run["args"], force=args.force)


if __name__ == "__main__":
    main()
