# -*- coding: utf-8 -*-
"""Build dense-LiDAR Table 4 clustering comparison markdown."""

import csv
import math
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table4_dense_20260729"
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
        "label": "sgcp",
        "display": "SGCP",
        "source": "proposed",
        "clustering": "potential_verified_cov_coalition_game",
        "log": SGCP_RUN / "run.out",
        "trace": SGCP_RUN / "trace.csv",
    },
    {
        "label": "random_balanced",
        "display": "Random balanced",
        "source": "heuristic",
        "clustering": "random_balanced",
        "log": ARTIFACT / "random_balanced.log",
        "trace": ARTIFACT / "random_balanced_trace.csv",
    },
    {
        "label": "distance_greedy",
        "display": "Distance-greedy",
        "source": "heuristic",
        "clustering": "distance_greedy",
        "log": ARTIFACT / "distance_greedy.log",
        "trace": ARTIFACT / "distance_greedy_trace.csv",
    },
    {
        "label": "density_greedy",
        "display": "Density/quality-greedy",
        "source": "heuristic",
        "clustering": "density_greedy_cluster",
        "log": ARTIFACT / "density_greedy_cluster.log",
        "trace": ARTIFACT / "density_greedy_cluster_trace.csv",
    },
    {
        "label": "seac_social",
        "display": "SeAC-inspired",
        "source": "paper baseline",
        "clustering": "seac_social_adaptive",
        "log": ARTIFACT / "seac_social_adaptive.log",
        "trace": ARTIFACT / "seac_social_adaptive_trace.csv",
    },
    {
        "label": "hho_vanet",
        "display": "HHOCNET-inspired",
        "source": "paper baseline",
        "clustering": "hho_vanet",
        "log": ARTIFACT / "hho_vanet.log",
        "trace": ARTIFACT / "hho_vanet_trace.csv",
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
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
        "receiver_policy": most_common(row.get("receiver_policy", "") for row in rows),
        "resource_allocation": most_common(row.get("resource_allocation", "") for row in rows),
        "clustering": most_common(row.get("clustering", "") for row in rows),
        "mean_link_time_ms": sum(times) / max(len(times), 1),
        "p95_link_time_ms": p95,
        "max_link_time_ms": max(times) if times else 0.0,
    }


def build_rows():
    rows = []
    for run in RUNS:
        log = parse_log(run["log"])
        trace = parse_trace(run["trace"])
        duration_s = max(trace["frames"] * FRAME_INTERVAL_S, 1e-9)
        raw_mbps = log["payload_bytes"] * 8.0 / duration_s / 1e6
        rows.append({
            "label": run["label"],
            "display": run["display"],
            "baseline_type": run["source"],
            "clustering": trace["clustering"] or run["clustering"],
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
            "trace_path": str(run["trace"]),
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_compute_profile(metric_csv):
    output_csv = ARTIFACT / "table4_dense_compute_20260729.csv"
    output_md = ARTIFACT / "table4_dense_compute_20260729.md"
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
        "# Dense-LiDAR Table 4. Clustering Baselines under the SGCP Protocol",
        "",
        "Protocol: `v2xp_cluster_carla_dense`, 20 CAVs, 41 frames (`000060-000140`), attentive-derived early detector, 40 MHz / 10 target subchannels, NS3 estimator `tb_size=899 B`, `slot=0.5 ms`, `symbols=12`, `mcs=28`, 60 ms data-plane deadline, `cov_potential_game` C->V raw-LiDAR scheduler, all cluster heads as receivers, grid upload, inter-cluster box NMS. Only the clustering algorithm changes.",
        "",
        "| Method | Baseline type | Clustering | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg grids | P95 link time |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        comp = compute_rows[row["label"]]
        lines.append(
            "| {display} | {baseline_type} | {clustering} | {ap_03} | "
            "{ap_05} | {ap_07} | {raw} | {box} | {total} | {gflops} | "
            "{sources} | {grids} | {p95} ms |".format(
                display=row["display"],
                baseline_type=row["baseline_type"],
                clustering=row["clustering"],
                ap_03=row["ap_03"],
                ap_05=row["ap_05"],
                ap_07=row["ap_07"],
                raw=fmt(row["raw_lidar_mbps"]),
                box=fmt(row["box_mbps"]),
                total=fmt(row["total_mbps"]),
                gflops=fmt(comp["input_adjusted_detector_gflops_per_frame"]),
                sources=fmt(row["avg_source_cavs"]),
                grids=fmt(row["avg_selected_grids"]),
                p95=fmt(row["p95_link_time_ms"]),
            )
        )
    lines.extend([
        "",
        "Notes:",
        "- `Total Mbps = Raw Mbps + Box Mbps`; all rows include the same inter-cluster detection-box communication accounting.",
        "- This table is a clustering comparison, not a protocol-native baseline table. Resource scheduling, late fusion, checkpoint, channel estimator and communication accounting are fixed.",
        "- Dense LiDAR reduces the AP gap among clustering methods. SGCP keeps the best AP@0.7 among the tested clustering methods while using comparable detector compute.",
        "",
    ])
    return "\n".join(lines)


def main():
    rows = build_rows()
    metric_csv = ARTIFACT / "table4_dense_metrics_20260729.csv"
    write_csv(metric_csv, rows)
    compute_csv = run_compute_profile(metric_csv)
    markdown = build_markdown(rows, load_compute(compute_csv))
    for output in [
        ARTIFACT / "table4_dense_20260729.md",
        EXPERIMENT / "04_clustering_ablation.md",
    ]:
        output.write_text(markdown, encoding="utf-8")
        print(output)
    print(metric_csv)
    print(compute_csv)


if __name__ == "__main__":
    main()
