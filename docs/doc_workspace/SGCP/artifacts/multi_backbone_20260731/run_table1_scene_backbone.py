# -*- coding: utf-8 -*-
"""Run protocol-native Table 1 for a selected scene/backbone pair."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
DATASET_ROOT = Path(r"D:\Data\Carla")


def build_runs(n_max, sgcp_budget_mbps):
    return [
        {
            "name": "centralized_upper",
            "args": [
                "--sgcp-constrained",
                "--clustering", "all_in_one",
                "--sgcp-receiver-policy", "all-cluster-heads",
                "--sgcp-upload-mode", "full_cluster",
            ],
        },
        {
            "name": "no_collaboration",
            "args": [
                "--sgcp-constrained",
                "--clustering", "singleton",
                "--sgcp-receiver-policy", "all-cavs",
                "--sgcp-upload-mode", "head_only",
            ],
        },
        {
            "name": "pure_late",
            "args": [
                "--sgcp-constrained",
                "--clustering", "singleton",
                "--sgcp-receiver-policy", "all-cavs",
                "--sgcp-upload-mode", "head_only",
                "--sgcp-inter-cluster-late-fusion",
            ],
        },
        {
            "name": "pcs",
            "args": [
                "--sgcp-constrained",
                "--clustering", "singleton",
                "--resource-allocation", "fullperception_pcs",
                "--sgcp-receiver-policy", "all-cavs",
                "--sgcp-upload-mode", "grid",
                "--pcs-frame-deadline-ms", "60",
                "--pcs-blind-spot-min-division", "4",
                "--pcs-blind-spot-radius", "4",
                "--pcs-min-spot-grids", "128",
                "--pcs-communication-range-m", "35",
                "--max-senders-per-receiver", "2",
                "--rho-th", "2",
            ],
        },
        {
            "name": "edgecooper_pmax",
            "args": [
                "--selective-sharing-baseline", "edgecooper_global_pmax",
                "--clustering", "singleton",
                "--sgcp-receiver-policy", "all-cavs",
                "--sgcp-upload-mode", "grid",
                "--selective-member-budget", "3",
                "--selective-grid-budget", "117",
                "--selective-frame-deadline-ms", "60",
                "--edgecooper-global-comm-range-m", "35",
                "--edgecooper-pmax-density-cap-rho", "2",
                "--max-senders-per-receiver", "2",
                "--rho-th", "2",
            ],
        },
        {
            "name": "pacp_lidar",
            "args": [
                "--selective-sharing-baseline", "pacp_lidar",
                "--clustering", "singleton",
                "--sgcp-receiver-policy", "all-cavs",
                "--sgcp-upload-mode", "grid",
                "--selective-member-budget", "3",
                "--selective-grid-budget", "117",
                "--selective-frame-deadline-ms", "60",
                "--edgecooper-global-comm-range-m", "35",
                "--max-senders-per-receiver", "2",
                "--rho-th", "2",
            ],
        },
        {
            "name": "sgcp",
            "args": [
                "--sgcp-constrained",
                "--clustering", "potential_verified_cov_coalition_game",
                "--resource-allocation", "dynamic_cv",
                "--sgcp-receiver-policy", "all-cluster-heads",
                "--sgcp-upload-mode", "grid",
                "--sgcp-inter-cluster-late-fusion",
                "--n-max", str(n_max),
                "--rho-th", "2",
                "--head-rb-budget", "2",
                "--sgcp-frame-mbps-budget", str(sgcp_budget_mbps),
                "--upload-density-cap-rho", "2",
            ],
        },
    ]


def run_one(base_args, output_dir, name, extra_args, force=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / ("%s.out" % name)
    trace_path = output_dir / ("%s_trace.csv" % name)
    eval_path = output_dir / ("%s_eval_stats.csv" % name)
    if log_path.exists() and trace_path.exists() and not force:
        print("skip existing %s" % name)
        return
    cmd = [
        sys.executable,
        *base_args,
        *extra_args,
        "--sgcp-trace-output", str(trace_path),
        "--eval-stats-output", str(eval_path),
    ]
    print("running %s" % name, flush=True)
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
    print("done %s" % name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--coperception-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-max", type=int, required=True)
    parser.add_argument("--sgcp-budget-mbps", type=float, default=40.0)
    parser.add_argument("--cav-count", type=int, default=None)
    parser.add_argument("--local-preserving-output", action="store_true")
    parser.add_argument("--gt-scope", default="sample",
                        choices=["sample", "full-frame"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    base_args = [
        "-m", "opencda.tools.offline_inference",
        "--dataset-root", str(DATASET_ROOT),
        "--scenario-id", args.scenario_id,
        "--ego-cav-id", "1",
        "--max-frames", str(args.max_frames),
        "--fusion-method", "early",
        "--coperception-yaml", args.coperception_yaml,
        "--bandwidth-mhz", "40",
        "--num-channels", "10",
        "--channel-estimator", "ns3",
        "--ns3-tb-size-bytes", "899",
        "--ns3-slot-duration-ms", "0.5",
        "--ns3-subchannel-prbs", "10",
        "--ns3-symbols-per-slot", "12",
        "--ns3-mcs", "28",
        "--communication-deadline-ms", "60",
        "--gt-scope", args.gt_scope,
    ]
    if args.cav_count is not None:
        base_args.extend(["--cav-count", str(args.cav_count)])
    only = None
    if args.only:
        only = {item.strip() for item in args.only.split(",") if item.strip()}
    output_dir = Path(args.output_dir)
    local_preserving_runs = {"pcs", "edgecooper_pmax", "pacp_lidar"}
    for run in build_runs(args.n_max, args.sgcp_budget_mbps):
        if only is not None and run["name"] not in only:
            continue
        extra_args = list(run["args"])
        if args.local_preserving_output and run["name"] in local_preserving_runs:
            extra_args.append("--local-preserving-output")
        run_one(base_args, output_dir, run["name"], extra_args,
                force=args.force)


if __name__ == "__main__":
    main()
