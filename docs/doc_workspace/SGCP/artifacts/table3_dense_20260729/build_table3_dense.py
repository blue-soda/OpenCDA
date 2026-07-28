# -*- coding: utf-8 -*-
"""Build dense-LiDAR Table 3 scheduler comparison markdown."""

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table3_dense_20260729"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment-0729-dense-ver")
COMPUTE = EXPERIMENT / "compute"
SGCP_RUN = EXPERIMENT / "sgcp_40mhz_deadline60_budget60_nmax5_rho1_after_trim_41f"

FRAME_INTERVAL_S = 0.1
BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64

AP_PATTERN = re.compile(
    r"The Average Precision at IOU 0\.3 is\s+([0-9.]+),\s+"
    r"The Average Precision at IOU 0\.5 is\s+([0-9.]+),\s+"
    r"The Average Precision at IOU 0\.7 is\s+([0-9.]+)"
)
SUMMARY_PATTERN = re.compile(
    r"sgcp_summary\s+frames=(?P<trace_rows>\d+)\s+"
    r"avg_comm_bytes=(?P<avg_comm>[0-9.]+)\s+"
    r"total_comm_bytes=(?P<total_comm>\d+)\s+"
    r"avg_source_cavs=(?P<avg_sources>[0-9.]+)\s+"
    r"avg_selected_grids=(?P<avg_grids>[0-9.]+)"
)


RUNS = [
    {
        "label": "SGCP",
        "display": "SGCP",
        "scheduler": "cov_potential_game",
        "log": SGCP_RUN / "run.out",
        "trace": SGCP_RUN / "trace.csv",
        "main": True,
    },
    {
        "label": "Cluster-head late only",
        "display": "Cluster-head late only",
        "scheduler": "local_detection_head_only",
        "log": ARTIFACT / "head_only_k2.log",
        "trace": ARTIFACT / "head_only_k2_trace.csv",
        "main": True,
    },
    {
        "label": "FullPerception-PCS",
        "display": "FullPerception-PCS",
        "scheduler": "fullperception_pcs",
        "log": ARTIFACT / "pcs_k2.log",
        "trace": ARTIFACT / "pcs_k2_trace.csv",
        "main": True,
    },
    {
        "label": "Random budget",
        "display": "Random budget",
        "scheduler": "selective_random",
        "log": ARTIFACT / "random_k2.log",
        "trace": ARTIFACT / "random_k2_trace.csv",
        "main": True,
    },
    {
        "label": "Density greedy",
        "display": "Density greedy",
        "scheduler": "selective_density",
        "log": ARTIFACT / "density_k2.log",
        "trace": ARTIFACT / "density_k2_trace.csv",
        "main": True,
    },
    {
        "label": "Link-aware density",
        "display": "Link-aware density",
        "scheduler": "selective_communication_aware",
        "log": ARTIFACT / "linkaware_k2.log",
        "trace": ARTIFACT / "linkaware_k2_trace.csv",
        "main": True,
    },
    {
        "label": "PACP-LiDAR",
        "display": "PACP-LiDAR",
        "scheduler": "selective_pacp_lidar",
        "log": ARTIFACT / "pacp_lidar_k2.log",
        "trace": ARTIFACT / "pacp_lidar_k2_trace.csv",
        "main": True,
    },
    {
        "label": "EdgeCooper-HD",
        "display": "EdgeCooper-HD",
        "scheduler": "selective_edgecooper_global_hd",
        "log": ARTIFACT / "edgecooper_hd_k2.log",
        "trace": ARTIFACT / "edgecooper_hd_k2_trace.csv",
        "main": True,
    },
]


def safe_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def most_common(values):
    values = [value for value in values if value not in ("", None)]
    if not values:
        return ""
    return max(set(values), key=values.count)


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    aps = AP_PATTERN.findall(text)
    summaries = SUMMARY_PATTERN.findall(text)
    if not aps or not summaries:
        raise RuntimeError("Missing AP or sgcp_summary in %s" % path)
    ap_03, ap_05, ap_07 = aps[-1]
    trace_rows, avg_comm, total_comm, avg_sources, avg_grids = summaries[-1]
    return {
        "ap_03": float(ap_03),
        "ap_05": float(ap_05),
        "ap_07": float(ap_07),
        "trace_rows": int(trace_rows),
        "payload_bytes": int(total_comm),
        "avg_source_cavs": float(avg_sources),
        "avg_selected_grids": float(avg_grids),
    }


def parse_trace(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    timestamps = sorted({row.get("timestamp", "") for row in rows if row.get("timestamp", "")})
    by_sample = {}
    times = []
    for row in rows:
        key = (row.get("timestamp", ""), row.get("receiver_id", ""))
        if key[0] and key[1]:
            by_sample[key] = max(by_sample.get(key, 0), safe_int(row.get("pred_boxes")))
        times.append(safe_float(row.get("frame_comm_time_ms")))
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values()
        if boxes > 0
    )
    duration_s = max(len(timestamps) * FRAME_INTERVAL_S, 1e-9)
    times_sorted = sorted(times)
    p95 = times_sorted[math.ceil(0.95 * len(times_sorted)) - 1] if times_sorted else 0.0
    return {
        "frames": len(timestamps),
        "box_bytes": box_bytes,
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
        "receiver_policy": most_common(row.get("receiver_policy", "") for row in rows),
        "clustering": most_common(row.get("clustering", "") for row in rows),
        "max_senders_per_receiver": most_common(row.get("max_senders_per_receiver", "") for row in rows),
        "mean_link_time_ms": sum(times) / max(len(times), 1),
        "p95_link_time_ms": p95,
        "max_link_time_ms": max(times) if times else 0.0,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_metric_rows():
    rows = []
    for run in RUNS:
        log = parse_log(run["log"])
        trace = parse_trace(run["trace"])
        duration_s = max(trace["frames"] * FRAME_INTERVAL_S, 1e-9)
        raw_mbps = log["payload_bytes"] * 8.0 / duration_s / 1e6
        rows.append({
            "label": run["label"],
            "display": run["display"],
            "scheduler": run["scheduler"],
            "late_fusion": "inter_cluster_nms",
            "clustering": trace["clustering"],
            "receiver_policy": trace["receiver_policy"],
            "ap_03": "%.2f" % log["ap_03"],
            "ap_05": "%.2f" % log["ap_05"],
            "ap_07": "%.2f" % log["ap_07"],
            "raw_lidar_mbps": "%.6f" % raw_mbps,
            "box_mbps": "%.6f" % trace["box_mbps"],
            "total_mbps": "%.6f" % (raw_mbps + trace["box_mbps"]),
            "avg_source_cavs": "%.2f" % log["avg_source_cavs"],
            "avg_selected_grids": "%.2f" % log["avg_selected_grids"],
            "mean_link_time_ms": "%.2f" % trace["mean_link_time_ms"],
            "p95_link_time_ms": "%.2f" % trace["p95_link_time_ms"],
            "max_link_time_ms": "%.2f" % trace["max_link_time_ms"],
            "max_senders_per_receiver": trace["max_senders_per_receiver"],
            "trace_path": str(run["trace"]),
        })
    return rows


def run_compute_profile(metric_csv):
    output_csv = ARTIFACT / "table3_dense_compute_20260729.csv"
    output_md = ARTIFACT / "table3_dense_compute_20260729.md"
    cmd = [
        sys.executable,
        "-m", "opencda.tools.sgcp_compute_profile",
        "--metrics-csv", str(metric_csv),
        "--calibration-json", str(COMPUTE / "dense_singleton_forward_flops.json"),
        "--dense-calibration-json", str(COMPUTE / "dense_full20_forward_flops.json"),
        "--output-csv", str(output_csv),
        "--summary-md", str(output_md),
    ]
    for run in RUNS:
        cmd.extend(["--method", "%s=%s" % (run["label"], run["trace"])])
    subprocess.run(cmd, cwd=str(REPO), check=True)
    return output_csv


def load_compute(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["label"]: row for row in csv.DictReader(stream)}


def fmt(value):
    return "%.2f" % safe_float(value)


def build_markdown(rows, compute_rows):
    lines = [
        "# Dense-LiDAR Table 3. SGCP-Compatible Scheduler Comparison",
        "",
        "Protocol: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, `potential_verified_cov_coalition_game` clustering, all cluster heads as receivers, grid raw-LiDAR upload, and inter-cluster box NMS. Only the scheduler/protocol changes. SGCP uses `N_max=5`, `rho_th=1`, and a 60 Mbps raw-LiDAR frame cap with per-link deadline trimming.",
        "",
        "| Method | Late fusion | Clustering | Scheduler/protocol | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids | P95 link time |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        comp = compute_rows[row["label"]]
        lines.append(
            "| {display} | {late_fusion} | {clustering} | {scheduler} | "
            "{ap_03} | {ap_05} | {ap_07} | {raw} | {box} | {total} | "
            "{gflops} | {grids} | {p95} ms |".format(
                display=row["display"],
                late_fusion=row["late_fusion"],
                clustering=row["clustering"],
                scheduler=row["scheduler"],
                ap_03=row["ap_03"],
                ap_05=row["ap_05"],
                ap_07=row["ap_07"],
                raw=fmt(row["raw_lidar_mbps"]),
                box=fmt(row["box_mbps"]),
                total=fmt(row["total_mbps"]),
                gflops=fmt(comp["input_adjusted_detector_gflops_per_frame"]),
                grids=fmt(row["avg_selected_grids"]),
                p95=fmt(row["p95_link_time_ms"]),
            )
        )
    lines.extend([
        "",
        "Notes:",
        "- This is a scheduler comparison under the same SGCP-compatible clustering and late-aggregation scaffold, not a protocol-native baseline table.",
        "- `Raw Mbps` is scheduled raw-LiDAR payload; `Box Mbps` is inter-cluster detection-box aggregation payload; `Total Mbps = Raw Mbps + Box Mbps`.",
        "- Dense LiDAR makes the cluster-head-only lower reference much stronger than in the sparse package. The scheduler table should therefore be interpreted together with GFLOPs: SGCP improves AP over cluster-head late only while keeping detector compute far below all-CAV pure late.",
        "- Baseline scheduler rows use K=2 receiver-side concurrent inbound links where applicable, aligned with SGCP receiver capability.",
        "",
    ])
    return "\n".join(lines)


def main():
    rows = build_metric_rows()
    metric_csv = ARTIFACT / "table3_dense_metrics_20260729.csv"
    write_csv(metric_csv, rows)
    compute_csv = run_compute_profile(metric_csv)
    markdown = build_markdown(rows, load_compute(compute_csv))
    for output in [
        ARTIFACT / "table3_dense_20260729.md",
        EXPERIMENT / "03_scheduler_comparison.md",
    ]:
        output.write_text(markdown, encoding="utf-8")
        print(output)
    print(metric_csv)
    print(compute_csv)


if __name__ == "__main__":
    main()
