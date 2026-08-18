# -*- coding: utf-8 -*-
"""Run SGCP hybrid scheduler CAV-count/Nmax sweep on dense Town03.

Runs the paired configurations (N=5,N_max=2), (N=10,N_max=3), and
(N=15,N_max=4).  Old dynamic_cv and pure-late references are parsed from the
earlier dense CAV sweep artifact.
"""

import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = (
    REPO / "docs/doc_workspace/SGCP/artifacts"
    / "hybrid_cav_nmax_sweep_20260801")
OLD_ARTIFACT = (
    REPO / "docs/doc_workspace/SGCP/artifacts"
    / "dense_cav_sweep_fullgt_20260801")
DATASET_ROOT = r"D:\Data\Carla"
SCENARIO_ID = "2026_07_29_02_32_08"
COPERCEPTION_YAML = (
    "docs\\doc_workspace\\SGCP\\artifacts\\early_from_late_checkpoint_20260719"
    "\\enable_coperception_early_from_attentive.yaml")

NMAX_PAIRS = [(5, 2), (10, 3), (15, 4)]
N_VALUES = [item[0] for item in NMAX_PAIRS]
FRAME_COUNT = 41
FRAME_INTERVAL_S = 0.1
BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64


def command_for(n_cavs, nmax, out_dir):
    return [
        "conda", "run", "--no-capture-output", "-n", "opencda",
        "python", "-m", "opencda.tools.offline_inference",
        "--dataset-root", DATASET_ROOT,
        "--scenario-id", SCENARIO_ID,
        "--ego-cav-id", "1",
        "--max-frames", str(FRAME_COUNT),
        "--fusion-method", "early",
        "--coperception-yaml", COPERCEPTION_YAML,
        "--bandwidth-mhz", "40",
        "--num-channels", "10",
        "--channel-estimator", "ns3",
        "--ns3-tb-size-bytes", "899",
        "--ns3-slot-duration-ms", "0.5",
        "--ns3-subchannel-prbs", "10",
        "--ns3-symbols-per-slot", "12",
        "--ns3-mcs", "28",
        "--communication-deadline-ms", "60",
        "--gt-scope", "full-frame",
        "--cav-count", str(n_cavs),
        "--sgcp-constrained",
        "--clustering", "potential_verified_cov_coalition_game",
        "--resource-allocation", "hybrid_round_robin_dynamic_marginal",
        "--sgcp-receiver-policy", "all-cluster-heads",
        "--sgcp-upload-mode", "grid",
        "--sgcp-inter-cluster-late-fusion",
        "--n-max", str(nmax),
        "--rho-th", "2",
        "--head-rb-budget", "2",
        "--sgcp-frame-mbps-budget", "40.0",
        "--upload-density-cap-rho", "2",
        "--sgcp-trace-output", str(out_dir / "trace.csv"),
        "--eval-stats-output", str(out_dir / "eval_stats.csv"),
    ]


def run_missing():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    for n_cavs, nmax in NMAX_PAIRS:
        out_dir = ARTIFACT / ("N%d_nmax%d" % (n_cavs, nmax))
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.out"
        if log_path.exists() and "Evaluate final average precision" in (
                log_path.read_text(encoding="utf-8", errors="ignore")):
            print("skip completed", out_dir)
            continue
        cmd = command_for(n_cavs, nmax, out_dir)
        (out_dir / "command.txt").write_text(
            " ".join(cmd) + "\n", encoding="utf-8")
        print("running", out_dir)
        with log_path.open("w", encoding="utf-8", errors="replace") as out:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "command failed for %s with %s" % (out_dir, proc.returncode))


def parse_ap(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"IOU 0\.3 is\s+([0-9.]+).*?IOU 0\.5 is\s+([0-9.]+).*?"
        r"IOU 0\.7 is\s+([0-9.]+)",
        text,
        re.S)
    if not matches:
        return "", "", ""
    return tuple(float(item) for item in matches[-1])


def parse_summary(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"sgcp_summary frames=(\d+) avg_comm_bytes=([0-9.]+)\s+"
        r"total_comm_bytes=(\d+) avg_source_cavs=([0-9.]+)\s+"
        r"avg_selected_grids=([0-9.]+)",
        text)
    if not matches:
        return {
            "samples": 0,
            "raw_bytes": 0,
            "avg_source_cavs": 0.0,
            "avg_selected_grids": 0.0,
        }
    frames, _, total, avg_sources, avg_grids = matches[-1]
    return {
        "samples": int(frames),
        "raw_bytes": int(total),
        "avg_source_cavs": float(avg_sources),
        "avg_selected_grids": float(avg_grids),
    }


def trace_box_bytes(trace_path):
    if not trace_path.exists():
        return 0, 0.0, 0.0
    with trace_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_sample = {}
    frame_times = []
    timestamps = sorted({row.get("timestamp", "") for row in rows
                         if row.get("timestamp", "")})
    for ts in timestamps:
        times = [
            float(row.get("frame_comm_time_ms") or 0.0)
            for row in rows if row.get("timestamp") == ts
        ]
        if times:
            frame_times.append(max(times))
    for row in rows:
        ts = row.get("timestamp", "")
        rid = row.get("receiver_id", "")
        if not ts or not rid:
            continue
        boxes = int(float(row.get("pred_boxes") or 0))
        by_sample[(ts, rid)] = max(by_sample.get((ts, rid), 0), boxes)
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values() if boxes > 0)
    p95 = 0.0
    if frame_times:
        frame_times = sorted(frame_times)
        p95 = frame_times[math.ceil(0.95 * len(frame_times)) - 1]
    return box_bytes, p95, max(frame_times) if frame_times else 0.0


def build_row(method, n_cavs, nmax, log_path, trace_path):
    ap03, ap05, ap07 = parse_ap(log_path)
    summary = parse_summary(log_path)
    box_bytes, p95_time, max_time = trace_box_bytes(trace_path)
    duration_s = FRAME_COUNT * FRAME_INTERVAL_S
    raw_mbps = summary["raw_bytes"] * 8.0 / duration_s / 1e6
    box_mbps = box_bytes * 8.0 / duration_s / 1e6
    return {
        "method": method,
        "n_cavs": n_cavs,
        "n_max": "" if nmax is None else nmax,
        "ap_03": ap03,
        "ap_05": ap05,
        "ap_07": ap07,
        "raw_mbps": raw_mbps,
        "box_mbps": box_mbps,
        "total_mbps": raw_mbps + box_mbps,
        "samples": summary["samples"],
        "avg_source_cavs": summary["avg_source_cavs"],
        "avg_selected_grids": summary["avg_selected_grids"],
        "p95_data_time_ms": p95_time,
        "max_data_time_ms": max_time,
        "log": str(log_path),
        "trace": str(trace_path),
    }


def summarize():
    rows = []
    for n_cavs in N_VALUES:
        old_dir = OLD_ARTIFACT / ("N%d" % n_cavs)
        rows.append(build_row(
            "old_dynamic_cv",
            n_cavs,
            None,
            old_dir / "sgcp.out",
            old_dir / "sgcp_trace.csv"))
        rows.append(build_row(
            "pure_late",
            n_cavs,
            None,
            old_dir / "pure_late.out",
            old_dir / "pure_late_trace.csv"))
        pair_nmax = dict(NMAX_PAIRS)[n_cavs]
        out_dir = ARTIFACT / ("N%d_nmax%d" % (n_cavs, pair_nmax))
        rows.append(build_row(
            "hybrid_round_robin_dynamic_marginal",
            n_cavs,
            pair_nmax,
            out_dir / "run.out",
            out_dir / "trace.csv"))
    rows.sort(key=lambda row: (
        int(row["n_cavs"]),
        {"pure_late": 0, "old_dynamic_cv": 1}.get(row["method"], 2),
        int(row["n_max"] or 0)))

    fieldnames = [
        "method", "n_cavs", "n_max", "ap_03", "ap_05", "ap_07",
        "raw_mbps", "box_mbps", "total_mbps", "samples",
        "avg_source_cavs", "avg_selected_grids", "p95_data_time_ms",
        "max_data_time_ms", "log", "trace",
    ]
    csv_path = ARTIFACT / "hybrid_cav_nmax_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = ARTIFACT / "hybrid_cav_nmax_sweep.md"
    pair_text = ", ".join(
        "`(%d,%d)`" % (n_cavs, nmax) for n_cavs, nmax in NMAX_PAIRS)
    lines = [
        "# Hybrid Scheduler CAV Count / Nmax Sweep",
        "",
        "Protocol: dense Town03 `2026_07_29_02_32_08`, 41 frames "
        "`000060-000140`, attentive-to-early checkpoint, full-frame GT, "
        "40 MHz / 10 target subchannels, NS3 estimator `tb_size=899`, "
        "`slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, "
        "`rho_th=2`, density cap `rho=2`, raw-LiDAR admission budget 40 Mbps, "
        "`head_rb_budget=2`, inter-cluster box NMS. New SGCP rows use paired "
        "`(N,N_max)` settings %s and " % pair_text +
        "`hybrid_round_robin_dynamic_marginal`; `old_dynamic_cv` and "
        "`pure_late` are parsed from `dense_cav_sweep_fullgt_20260801`.",
        "",
        "| Method | N CAVs | N_max | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | Samples | Avg source CAVs | Avg selected grids | P95 data time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {n_cavs} | {n_max} | {ap_03:.2f} | {ap_05:.2f} | "
            "{ap_07:.2f} | {raw_mbps:.2f} | {box_mbps:.2f} | "
            "{total_mbps:.2f} | {samples} | {avg_source_cavs:.2f} | "
            "{avg_selected_grids:.2f} | {p95_data_time_ms:.2f} |".format(
                **row))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


def main():
    run_missing()
    summarize()


if __name__ == "__main__":
    main()
