# -*- coding: utf-8 -*-
"""Run paper-facing SGCP Table 4 clustering comparison.

Only the clustering method changes.  All rows use the same attentive detector,
formal C/V scheduler, inter-cluster box NMS and NS3-calibrated channel
estimator.
"""

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table4_clustering_cv_20260724"
DATASET_ROOT = Path(r"D:\Data\Carla")
SCENARIO_ID = "2026_07_15_01_26_56"
YAML = REPO / (
    "docs/doc_workspace/SGCP/artifacts/"
    "early_from_late_checkpoint_20260719/"
    "enable_coperception_early_from_attentive.yaml"
)


BASE_ARGS = [
    "-m", "opencda.tools.offline_inference",
    "--dataset-root", str(DATASET_ROOT),
    "--scenario-id", SCENARIO_ID,
    "--ego-cav-id", "1",
    "--max-frames", "41",
    "--fusion-method", "early",
    "--coperception-yaml", str(YAML),
    "--sgcp-constrained",
    "--resource-allocation", "cov_potential_game",
    "--sgcp-receiver-policy", "all-cluster-heads",
    "--sgcp-upload-mode", "grid",
    "--sgcp-inter-cluster-late-fusion",
    "--sgcp-grid-selection-mode", "utility",
    "--sgcp-grid-score-mode", "utility",
    "--bandwidth-mhz", "40",
    "--num-channels", "10",
    "--channel-estimator", "ns3",
    "--ns3-tb-size-bytes", "899",
    "--ns3-slot-duration-ms", "0.5",
    "--ns3-subchannel-prbs", "10",
    "--ns3-symbols-per-slot", "12",
    "--ns3-mcs", "28",
    "--n-max", "4",
    "--rho-th", "3",
    "--head-rb-budget", "2",
    "--sgcp-frame-mbps-budget", "200",
]


RUNS = [
    "random_balanced",
    "distance_greedy",
    "density_greedy_cluster",
    "seac_social_adaptive",
    "hho_vanet",
]


def run_one(clustering, force=False):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT / ("%s.log" % clustering)
    trace_path = ARTIFACT / ("%s_trace.csv" % clustering)
    eval_path = ARTIFACT / ("%s_eval_stats.csv" % clustering)
    if log_path.exists() and trace_path.exists() and not force:
        print("skip existing %s" % clustering)
        return
    cmd = [
        sys.executable,
        *BASE_ARGS,
        "--clustering", clustering,
        "--sgcp-trace-output", str(trace_path),
        "--eval-stats-output", str(eval_path),
    ]
    print("running %s" % clustering)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: %s\n\n" % " ".join(cmd))
        log.flush()
        subprocess.run(cmd, cwd=str(REPO), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print("done %s -> %s" % (clustering, log_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    only = None
    if args.only:
        only = set(item.strip() for item in args.only.split(",") if item.strip())
    for clustering in RUNS:
        if only is not None and clustering not in only:
            continue
        run_one(clustering, force=args.force)


if __name__ == "__main__":
    main()
