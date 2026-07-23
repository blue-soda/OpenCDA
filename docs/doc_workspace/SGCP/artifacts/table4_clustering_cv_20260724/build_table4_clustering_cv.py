# -*- coding: utf-8 -*-
"""Build the clean Table 4 clustering comparison package."""

import csv
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table4_clustering_cv_20260724"
PARAM = REPO / "docs/doc_workspace/SGCP/artifacts/parameter_sensitivity_cv_20260723"
COMPUTE = REPO / "docs/doc_workspace/SGCP/artifacts/compute_profile_20260722"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment")

BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1


RUNS = [
    {
        "label": "sgcp_cv",
        "display": "SGCP-CV (ours)",
        "clustering": "cov_coalition_game",
        "source": "proposed",
        "log": PARAM / "mbps200.log",
        "trace": PARAM / "mbps200_trace.csv",
        "note": (
            "Proposed V-only coalition formation plus C->V potential-game "
            "scheduler; selected main operating point from Raw LiDAR Mbps "
            "Budget sweep."
        ),
    },
    {
        "label": "random_balanced",
        "display": "Random balanced",
        "clustering": "random_balanced",
        "source": "heuristic",
        "log": ARTIFACT / "random_balanced.log",
        "trace": ARTIFACT / "random_balanced_trace.csv",
        "note": "Deterministic random balanced clusters.",
    },
    {
        "label": "distance_greedy",
        "display": "Distance-greedy",
        "clustering": "distance_greedy",
        "source": "heuristic",
        "log": ARTIFACT / "distance_greedy.log",
        "trace": ARTIFACT / "distance_greedy_trace.csv",
        "note": "Greedy proximity clusters with center-nearest head election.",
    },
    {
        "label": "density_greedy",
        "display": "Density/quality-greedy",
        "clustering": "density_greedy_cluster",
        "source": "heuristic",
        "log": ARTIFACT / "density_greedy_cluster.log",
        "trace": ARTIFACT / "density_greedy_cluster_trace.csv",
        "note": "Greedy sensing-density and incremental-coverage clusters.",
    },
    {
        "label": "seac_social",
        "display": "SeAC-inspired",
        "clustering": "seac_social_adaptive",
        "source": "paper baseline",
        "log": ARTIFACT / "seac_social_adaptive.log",
        "trace": ARTIFACT / "seac_social_adaptive_trace.csv",
        "note": (
            "Adapted from SeAC, IEEE T-ITS 2023: SDN/social information is "
            "mapped to same-frame direction, relative speed, distance and "
            "sensing-field overlap because the CARLA dump has no route/social "
            "history."
        ),
    },
    {
        "label": "hho_vanet",
        "display": "HHOCNET-inspired",
        "clustering": "hho_vanet",
        "source": "paper baseline",
        "log": ARTIFACT / "hho_vanet.log",
        "trace": ARTIFACT / "hho_vanet_trace.csv",
        "note": (
            "Adapted from HHO-based VANET clustering, IEEE T-ITS 2023: "
            "multi-start Harris-Hawks-style partition search over proximity, "
            "relative mobility and sensing coverage."
        ),
    },
]

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


def safe_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def safe_float(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def most_common(values):
    values = [value for value in values if value not in ("", None)]
    if not values:
        return ""
    return max(set(values), key=values.count)


def parse_log(path):
    text = path.read_text(errors="replace")
    ap_matches = AP_PATTERN.findall(text)
    summary_matches = SUMMARY_PATTERN.findall(text)
    if not ap_matches or not summary_matches:
        raise RuntimeError("Missing AP or sgcp_summary in %s" % path)
    ap_03, ap_05, ap_07 = ap_matches[-1]
    trace_rows, avg_comm, total_comm, avg_sources, avg_grids = (
        summary_matches[-1])
    return {
        "ap_03": ap_03,
        "ap_05": ap_05,
        "ap_07": ap_07,
        "trace_rows": trace_rows,
        "avg_comm_bytes": avg_comm,
        "payload_bytes": total_comm,
        "avg_source_cavs": avg_sources,
        "avg_selected_grids": avg_grids,
    }


def parse_trace(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Empty trace: %s" % path)
    timestamps = sorted({row.get("timestamp", "") for row in rows
                         if row.get("timestamp", "")})
    by_sample = {}
    for row in rows:
        timestamp = row.get("timestamp", "")
        receiver = row.get("receiver_id", "")
        if not timestamp or not receiver:
            continue
        key = (timestamp, receiver)
        by_sample[key] = max(by_sample.get(key, 0),
                             safe_int(row.get("pred_boxes")))
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values()
        if boxes > 0)
    return {
        "timestamps": len(timestamps),
        "box_bytes": box_bytes,
        "receiver_policy": most_common(
            [row.get("receiver_policy", "") for row in rows]),
        "num_channels": most_common(
            [row.get("num_channels", "") for row in rows]),
        "bandwidth_mhz": most_common(
            [row.get("bandwidth_mhz", "") for row in rows]),
        "channel_estimator": most_common(
            [row.get("channel_estimator", "") for row in rows]),
        "ns3_tb_size_bytes": most_common(
            [row.get("ns3_tb_size_bytes", "") for row in rows]),
        "ns3_symbols_per_slot": most_common(
            [row.get("ns3_symbols_per_slot", "") for row in rows]),
        "ns3_mcs": most_common([row.get("ns3_mcs", "") for row in rows]),
        "cluster_count": most_common(
            [row.get("cluster_count", "") for row in rows]),
    }


def build_metrics():
    rows = []
    for run in RUNS:
        log = parse_log(run["log"])
        trace = parse_trace(run["trace"])
        duration_s = max(trace["timestamps"] * FRAME_INTERVAL_S, 1e-9)
        raw_mbps = safe_int(log["payload_bytes"]) * 8.0 / duration_s / 1e6
        box_mbps = trace["box_bytes"] * 8.0 / duration_s / 1e6
        rows.append({
            **run,
            **log,
            **trace,
            "raw_lidar_mbps": raw_mbps,
            "box_mbps": box_mbps,
            "total_mbps": raw_mbps + box_mbps,
        })
    return rows


def write_metrics_csv(rows):
    output = ARTIFACT / "table4_clustering_cv_metrics_20260724.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "label", "display", "source", "clustering", "ap_03", "ap_05",
            "ap_07", "raw_lidar_mbps", "box_mbps", "total_mbps",
            "avg_source_cavs", "avg_selected_grids", "receiver_policy",
            "cluster_count", "num_channels", "bandwidth_mhz",
            "channel_estimator", "ns3_tb_size_bytes", "ns3_symbols_per_slot",
            "ns3_mcs", "note",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def run_compute_profile():
    output_csv = ARTIFACT / "table4_clustering_cv_compute_20260724.csv"
    output_md = ARTIFACT / "table4_clustering_cv_compute_20260724.md"
    cmd = [
        sys.executable,
        "-m", "opencda.tools.sgcp_compute_profile",
        "--calibration-json",
        str(COMPUTE / "attentive_singleton_forward_flops.json"),
        "--dense-calibration-json",
        str(COMPUTE / "attentive_full20_forward_flops.json"),
        "--output-csv", str(output_csv),
        "--summary-md", str(output_md),
    ]
    for run in RUNS:
        cmd.extend(["--method", "%s=%s" % (run["label"], run["trace"])])
    subprocess.run(cmd, cwd=str(REPO), check=True)
    return output_csv


def load_compute(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["label"]: row
            for row in csv.DictReader(stream)
        }


def fmt(value):
    return "%.2f" % safe_float(value)


def write_markdown(rows, compute_rows):
    lines = [
        "# Table 4. Clustering Baselines under the SGCP-CV Protocol",
        "",
        "Protocol: attentive detector, v2xp_cluster_carla 41-frame offline replay, 20 CAVs, 40 MHz / 10 target subchannels, NS3-calibrated estimator (`tb_size=899`, `symbols=12`, `mcs=28`), `cov_potential_game` C->V raw-LiDAR scheduler, all cluster heads as receivers, grid upload, inter-cluster box NMS. Only the clustering algorithm changes. SGCP-CV uses the selected main operating point from the Raw LiDAR Mbps Budget sweep; all non-SGCP rows are rerun under the same C/V scheduler and communication accounting.",
        "",
        "| Method | Baseline type | Clustering | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg grids |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        comp = compute_rows[row["label"]]
        lines.append(
            "| {display} | {source} | {clustering} | {ap03} | {ap05} | "
            "{ap07} | {raw} | {box} | {total} | {gflops} | "
            "{sources} | {grids} |".format(
                display=row["display"],
                source=row["source"],
                clustering=row["clustering"],
                ap03=row["ap_03"],
                ap05=row["ap_05"],
                ap07=row["ap_07"],
                raw=fmt(row["raw_lidar_mbps"]),
                box=fmt(row["box_mbps"]),
                total=fmt(row["total_mbps"]),
                gflops=fmt(
                    comp["input_adjusted_detector_gflops_per_frame"]),
                sources=fmt(row["avg_source_cavs"]),
                grids=fmt(row["avg_selected_grids"]),
            )
        )
    lines.extend([
        "",
        "Paper-baseline sources and adaptation:",
        "- SeAC-inspired maps Akbar et al., `SeAC: SDN-Enabled Adaptive Clustering Technique for Social-Aware Internet of Vehicles`, IEEE Transactions on Intelligent Transportation Systems, 24(5):4827-4835, 2023, DOI `10.1109/TITS.2023.3237321`, to CARLA-side direction, relative-speed, distance and sensing-overlap proxies.",
        "- HHOCNET-inspired maps Ali et al., `Harris Hawks Optimization-Based Clustering Algorithm for Vehicular Ad-Hoc Networks`, IEEE Transactions on Intelligent Transportation Systems, 24(6):5822-5841, 2023, DOI `10.1109/TITS.2023.3257484`, to deterministic multi-start partition search over proximity, relative mobility and sensing coverage.",
        "",
        "Notes:",
        "- `Total Mbps = Raw Mbps + Box Mbps`; all rows include the same inter-cluster detection-box communication accounting.",
        "- This table is a clustering comparison, not a protocol-native baseline table. Resource scheduling, late fusion, checkpoint, channel estimator and communication accounting are fixed.",
        "- `Avg source CAVs` and `Avg grids` are diagnostic columns kept in the experiment package to explain payload/AP changes; they can be removed from the camera-ready table if space is tight.",
        "",
    ])
    markdown = "\n".join(lines)
    for output in [
            ARTIFACT / "table4_clustering_cv_20260724.md",
            EXPERIMENT / "04_clustering_ablation.md"]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(output)


def main():
    rows = build_metrics()
    metrics_csv = write_metrics_csv(rows)
    compute_csv = run_compute_profile()
    write_markdown(rows, load_compute(compute_csv))
    print(metrics_csv)
    print(compute_csv)
    for row in rows:
        print("%s %s/%s/%s total=%.2f" % (
            row["display"],
            row["ap_03"],
            row["ap_05"],
            row["ap_07"],
            row["total_mbps"]))


if __name__ == "__main__":
    main()
