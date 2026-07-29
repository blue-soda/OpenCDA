# -*- coding: utf-8 -*-
"""Run dense-LiDAR SGCP rho_th sweep with a 120 Mbps frame budget.

This diagnostic intentionally does not pass --communication-deadline-ms, so it
matches the earlier dense budget20/40/60/80 sweeps: communication is constrained
by the raw-LiDAR Mbps frame cap, not by the 60 ms per-link deadline.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment-0729-dense-ver")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/dense_120mbps_rho_20260729"
DATASET_ROOT = Path(r"D:\Data\Carla")
SCENARIO_ID = "2026_07_29_02_32_08"

RHO_VALUES = ["0.5", "1", "2", "3", "5"]


BASE_ARGS = [
    "-m", "opencda.tools.offline_inference",
    "--dataset-root", str(DATASET_ROOT),
    "--scenario-id", SCENARIO_ID,
    "--ego-cav-id", "1",
    "--max-frames", "41",
    "--fusion-method", "early",
    "--sgcp-constrained",
    "--clustering", "potential_verified_cov_coalition_game",
    "--resource-allocation", "cov_potential_game",
    "--sgcp-receiver-policy", "all-cluster-heads",
    "--sgcp-inter-cluster-late-fusion",
    "--sgcp-upload-mode", "grid",
    "--sgcp-grid-selection-mode", "utility",
    "--sgcp-grid-score-mode", "utility",
    "--n-max", "5",
    "--num-channels", "10",
    "--bandwidth-mhz", "40",
    "--channel-estimator", "ns3",
    "--ns3-tb-size-bytes", "899",
    "--ns3-slot-duration-ms", "0.5",
    "--ns3-subchannel-prbs", "10",
    "--ns3-symbols-per-slot", "12",
    "--ns3-mcs", "28",
    "--sgcp-frame-mbps-budget", "120",
]


def stem_for_rho(rho):
    return "rho" + rho.replace(".", "p")


def parse_ap(log_text):
    pattern = re.compile(
        r"Average Precision at IOU 0\.3 is ([0-9.]+).*?"
        r"Average Precision at IOU 0\.5 is ([0-9.]+).*?"
        r"Average Precision at IOU 0\.7 is ([0-9.]+)",
        re.S,
    )
    match = pattern.search(log_text)
    if not match:
        return None, None, None
    return tuple(float(x) for x in match.groups())


def summarize_trace(trace_path):
    rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
    total_bytes = sum(float(row.get("communication_bytes") or 0) for row in rows)
    timestamps = sorted({row["timestamp"] for row in rows})
    mbps = total_bytes * 8.0 / (max(1, len(timestamps)) * 0.1) / 1e6
    source_counts = []
    grid_counts = []
    frame_times = []
    for row in rows:
        srcs = [x for x in (row.get("uploaded_source_ids") or "").split(";") if x]
        source_counts.append(len(srcs))
        try:
            grid_counts.append(sum(int(v) for v in re.findall(r":\s*(\d+)", row.get("selected_grid_counts_json") or "")))
        except ValueError:
            pass
        try:
            frame_times.append(float(row.get("frame_comm_time_ms") or 0))
        except ValueError:
            pass
    return {
        "trace_rows": len(rows),
        "unique_timestamps": len(timestamps),
        "raw_mbps": mbps,
        "avg_source_cavs": sum(source_counts) / len(source_counts) if source_counts else 0.0,
        "avg_selected_grids": sum(grid_counts) / len(grid_counts) if grid_counts else 0.0,
        "max_link_time_ms": max(frame_times) if frame_times else 0.0,
    }


def run_one(rho, deadline_ms=None, force=False):
    stem = stem_for_rho(rho)
    suffix = "deadline%s_" % str(deadline_ms).replace(".", "p") if deadline_ms else ""
    out_dir = EXPERIMENT / ("sgcp_40mhz_budget120_%snmax5_%s_41f" % (suffix, stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.out"
    trace_path = out_dir / "trace.csv"
    eval_path = out_dir / "eval_stats.csv"
    manifest_path = out_dir / "manifest.csv"
    if log_path.exists() and trace_path.exists() and not force:
        print("skip existing rho=%s" % rho)
    else:
        cmd = [
            sys.executable,
            *BASE_ARGS,
            "--rho-th", rho,
            "--sgcp-trace-output", str(trace_path),
            "--eval-stats-output", str(eval_path),
        ]
        if deadline_ms:
            cmd.extend(["--communication-deadline-ms", str(deadline_ms)])
        print("running rho=%s" % rho)
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
        print("done rho=%s" % rho)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    ap03, ap05, ap07 = parse_ap(log_text)
    stats = summarize_trace(trace_path)
    row = {
        "rho_th": rho,
        "communication_deadline_ms": "" if deadline_ms is None else deadline_ms,
        "ap_03": ap03,
        "ap_05": ap05,
        "ap_07": ap07,
        **stats,
        "log_path": str(log_path),
        "trace_path": str(trace_path),
    }
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def write_summary(rows, deadline_ms=None):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    suffix = "" if deadline_ms is None else "_deadline%s" % str(deadline_ms).replace(".", "p")
    csv_path = ARTIFACT / ("dense_120mbps_rho_summary%s.csv" % suffix)
    md_path = ARTIFACT / ("dense_120mbps_rho_summary%s.md" % suffix)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Dense 120 Mbps rho_th Diagnostic",
        "",
        "Protocol: dense 41-frame replay, SGCP, `N_max=5`, 40 MHz / 10ch, NS3 estimator `tb=899`, raw-LiDAR frame budget `120 Mbps`.",
        "",
        "Communication deadline: `%s`." % ("default scenario time_slot" if deadline_ms is None else str(deadline_ms) + " ms"),
        "",
        "| rho_th | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Avg source CAVs | Avg selected grids | Max link time |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {rho_th} | {ap_03:.2f} | {ap_05:.2f} | {ap_07:.2f} | "
            "{raw_mbps:.2f} | {avg_source_cavs:.2f} | {avg_selected_grids:.2f} | "
            "{max_link_time_ms:.2f} ms |".format(**row)
        )
    lines.extend([
        "",
        "This is a diagnostic upper-budget sweep. It should not replace the deadline-feasible dense main table unless a later NS3 timing protocol is also changed.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rho", default=",".join(RHO_VALUES))
    parser.add_argument("--deadline-ms", type=float, default=None)
    args = parser.parse_args()
    rows = []
    for rho in [x.strip() for x in args.rho.split(",") if x.strip()]:
        rows.append(run_one(rho, deadline_ms=args.deadline_ms, force=args.force))
    write_summary(rows, deadline_ms=args.deadline_ms)


if __name__ == "__main__":
    main()
