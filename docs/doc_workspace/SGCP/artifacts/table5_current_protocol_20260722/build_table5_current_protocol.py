# -*- coding: utf-8 -*-
"""Build current-protocol Table 5 CSV and Figure 7 from clustering reruns."""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table5_current_protocol_20260722"
TABLE2 = REPO / "docs/doc_workspace/SGCP/artifacts/table2_current_protocol_20260722"
TABLE3 = REPO / "docs/doc_workspace/SGCP/artifacts/table3_current_protocol_20260722"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment")
EXTERNAL_DATA = EXPERIMENT / "data"
EXTERNAL_FIG = EXPERIMENT / "figures"

BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1

RUNS = [
    ("Singleton pure late reference", TABLE2 / "pure_late_41f.log", TABLE2 / "pure_late_41f_trace.csv", "prediction_nms", "singleton", "local_detection", "Prediction-sharing reference; not a raw-LiDAR clustering baseline."),
    ("Random balanced clusters", ARTIFACT / "random_balanced_41f.log", ARTIFACT / "random_balanced_41f_trace.csv", "inter_cluster_nms", "random_balanced", "perception_aware_potential_game", "Deterministic random balanced clusters; same PAPG and late fusion as SGCP, replacing only clustering."),
    ("Distance-greedy clusters", ARTIFACT / "distance_greedy_41f.log", ARTIFACT / "distance_greedy_41f_trace.csv", "inter_cluster_nms", "distance_greedy", "perception_aware_potential_game", "Proximity/communication-style greedy clusters; same PAPG and late fusion as SGCP, replacing only clustering."),
    ("Mobility-stability greedy clusters", ARTIFACT / "mobility_stability_greedy_41f.log", ARTIFACT / "mobility_stability_greedy_41f_trace.csv", "inter_cluster_nms", "mobility_stability_greedy", "perception_aware_potential_game", "MASS/C-MASS-inspired mobility-aware grouping; same PAPG and late fusion as SGCP, replacing only clustering."),
    ("Density/quality-greedy clusters", ARTIFACT / "density_greedy_cluster_41f.log", ARTIFACT / "density_greedy_cluster_41f_trace.csv", "inter_cluster_nms", "density_greedy_cluster", "perception_aware_potential_game", "Sensing-density greedy clusters; same PAPG and late fusion as SGCP, replacing only clustering."),
    ("Fixed first-frame clusters", ARTIFACT / "fixed_first_frame_41f.log", ARTIFACT / "fixed_first_frame_41f_trace.csv", "inter_cluster_nms", "fixed_first_frame", "perception_aware_potential_game", "Same PAPG and late fusion as SGCP but cluster membership frozen from the first frame."),
    ("Dynamic coalition clusters (SGCP)", TABLE3 / "papg_41f.log", TABLE3 / "papg_41f_trace.csv", "inter_cluster_nms", "coalition_game", "perception_aware_potential_game", "Proposed dynamic coalition formation plus PAPG scheduling and inter-cluster NMS under strict current protocol."),
    ("All-in-one full raw sharing", TABLE2 / "one_cluster_full_early_41f.log", TABLE2 / "one_cluster_full_early_41f_trace.csv", "identity_single_cluster", "all_in_one", "full_cluster", "No clustering; full 20-CAV raw-sharing upper reference."),
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
        if value in ("", None):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


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
    trace_rows, avg_comm, total_comm, avg_sources, avg_grids = summary_matches[-1]
    return {
        "ap_03": ap_03,
        "ap_05": ap_05,
        "ap_07": ap_07,
        "trace_rows": trace_rows,
        "avg_comm_bytes_per_trace_row": avg_comm,
        "payload_bytes": total_comm,
        "avg_source_cavs": avg_sources,
        "avg_selected_grids": avg_grids,
    }


def parse_trace(path, late_fusion):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Empty trace: %s" % path)
    timestamps = sorted({row.get("timestamp", "") for row in rows if row.get("timestamp", "")})
    box_bytes = 0
    if late_fusion in {"inter_cluster_nms", "prediction_nms"}:
        by_sample = {}
        for row in rows:
            timestamp = row.get("timestamp", "")
            receiver = row.get("receiver_id", "")
            if not timestamp or not receiver:
                continue
            key = (timestamp, receiver)
            by_sample[key] = max(by_sample.get(key, 0), safe_int(row.get("pred_boxes")))
        box_bytes = sum(
            MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
            for boxes in by_sample.values()
            if boxes > 0
        )
    duration_s = max(len(timestamps) * FRAME_INTERVAL_S, 1e-9)
    return {
        "unique_timestamps": str(len(timestamps)),
        "receiver_policy": most_common([row.get("receiver_policy", "") for row in rows]),
        "resource_allocation": most_common([row.get("resource_allocation", "") for row in rows]),
        "clustering": most_common([row.get("clustering", "") for row in rows]),
        "upload_mode": most_common([row.get("upload_mode", "") for row in rows]),
        "num_channels": most_common([row.get("num_channels", "") for row in rows]),
        "bandwidth_mhz": most_common([row.get("bandwidth_mhz", "") for row in rows]),
        "communication_deadline_ms": most_common([row.get("communication_deadline_ms", "") for row in rows]),
        "channel_estimator": most_common([row.get("channel_estimator", "") for row in rows]),
        "ns3_tb_size_bytes": most_common([row.get("ns3_tb_size_bytes", "") for row in rows]),
        "ns3_slot_duration_ms": most_common([row.get("ns3_slot_duration_ms", "") for row in rows]),
        "ns3_subchannel_prbs": most_common([row.get("ns3_subchannel_prbs", "") for row in rows]),
        "ns3_symbols_per_slot": most_common([row.get("ns3_symbols_per_slot", "") for row in rows]),
        "ns3_mcs": most_common([row.get("ns3_mcs", "") for row in rows]),
        "cluster_count_mode": most_common([row.get("cluster_count", "") for row in rows]),
        "box_bytes": str(box_bytes),
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
    }


def build_rows():
    rows = []
    for variant, log_path, trace_path, late_fusion, clustering, resource_allocation, interpretation in RUNS:
        log = parse_log(log_path)
        trace = parse_trace(trace_path, late_fusion)
        raw_mbps = (
            safe_int(log["payload_bytes"]) * 8.0
            / (safe_int(trace["unique_timestamps"]) * FRAME_INTERVAL_S)
            / 1e6
        )
        rows.append({
            "variant": variant,
            "checkpoint": "attentive",
            "late_fusion": late_fusion,
            "clustering": clustering,
            "resource_allocation": resource_allocation,
            "ap_03": log["ap_03"],
            "ap_05": log["ap_05"],
            "ap_07": log["ap_07"],
            "mbps": "%.6f" % (raw_mbps + trace["box_mbps"]),
            "payload_bytes": log["payload_bytes"],
            "evaluated_samples": trace["unique_timestamps"],
            "trace_rows": log["trace_rows"],
            "receiver_policy": trace["receiver_policy"],
            "num_channels": trace["num_channels"],
            "bandwidth_mhz": trace["bandwidth_mhz"],
            "communication_deadline_ms": trace["communication_deadline_ms"],
            "channel_estimator": trace["channel_estimator"],
            "ns3_tb_size_bytes": trace["ns3_tb_size_bytes"],
            "ns3_slot_duration_ms": trace["ns3_slot_duration_ms"],
            "ns3_subchannel_prbs": trace["ns3_subchannel_prbs"],
            "ns3_symbols_per_slot": trace["ns3_symbols_per_slot"],
            "ns3_mcs": trace["ns3_mcs"],
            "traffic_type": "box/prediction overhead" if raw_mbps == 0 else "raw LiDAR + box",
            "interpretation": interpretation,
            "log_path": str(log_path.relative_to(REPO)),
            "trace_path": str(trace_path.relative_to(REPO)),
            "raw_lidar_mbps": "%.6f" % raw_mbps,
            "box_mbps": "%.6f" % trace["box_mbps"],
            "total_mbps": "%.6f" % (raw_mbps + trace["box_mbps"]),
            "box_sharing_mode": (
                "broadcast_boxes" if late_fusion in {"inter_cluster_nms", "prediction_nms"} else "none"
            ),
            "notes": (
                "P12 current-protocol Table5 diagnostic; 40MHz/10ch/60ms "
                "NS3-calibrated estimator. Not paper-facing final until the "
                "strict-budget PAPG operating point/narrative is resolved."
            ),
        })
    return rows


def write_csv(rows):
    outputs = [
        ARTIFACT / "table5_clustering_ablation_current_protocol_20260722.csv",
        EXTERNAL_DATA / "table5_clustering_ablation_current_protocol_20260722.csv",
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(output)


def write_figure(rows):
    EXTERNAL_FIG.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ["ap_03", "ap_05", "ap_07", "total_mbps"]:
        df[col] = df[col].astype(float)
    labels = [
        value.replace(" clusters", "").replace(" current-protocol", "")
        for value in df["variant"]
    ]
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - width, df["ap_03"], width, label="AP@0.3")
    ax.bar(x, df["ap_05"], width, label="AP@0.5")
    ax.bar(x + width, df["ap_07"], width, label="AP@0.7")
    for i, row in df.iterrows():
        ax.text(i, 0.025, "%.1f Mbps" % row["total_mbps"], ha="center",
                va="bottom", rotation=90, fontsize=7, color="0.25")
    ax.set_title("Current-protocol clustering diagnostic (40MHz/10ch/60ms)")
    ax.set_ylabel("Aggregate AP")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    plt.tight_layout()
    for suffix in ["png", "pdf"]:
        output = EXTERNAL_FIG / (
            "figure7_clustering_ablation_current_protocol_20260722.%s" % suffix
        )
        fig.savefig(output, dpi=220 if suffix == "png" else None)
        print(output)
    plt.close(fig)


def main():
    rows = build_rows()
    write_csv(rows)
    write_figure(rows)
    for row in rows:
        print(
            "%s %s/%s/%s total=%s" % (
                row["variant"], row["ap_03"], row["ap_05"], row["ap_07"],
                row["total_mbps"]
            )
        )


if __name__ == "__main__":
    main()
