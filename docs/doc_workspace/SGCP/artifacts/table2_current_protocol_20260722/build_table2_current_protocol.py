# -*- coding: utf-8 -*-
"""Build current-protocol Table 2 CSV and Figure 3 from fusion reruns."""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table2_current_protocol_20260722"
TABLE3 = REPO / "docs/doc_workspace/SGCP/artifacts/table3_current_protocol_20260722"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment")
EXTERNAL_DATA = EXPERIMENT / "data"
EXTERNAL_FIG = EXPERIMENT / "figures"

BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1

RUNS = [
    ("HeadOnly_current_protocol", ARTIFACT / "head_only_41f.log", ARTIFACT / "head_only_41f_trace.csv", "inter_cluster_nms"),
    ("PureLate_current_protocol", ARTIFACT / "pure_late_41f.log", ARTIFACT / "pure_late_41f_trace.csv", "prediction_nms"),
    ("OneClusterEarlyOnly_current_protocol", ARTIFACT / "one_cluster_full_early_41f.log", ARTIFACT / "one_cluster_full_early_41f_trace.csv", "none"),
    ("ClusteredEarlyOnly_current_protocol", ARTIFACT / "clustered_early_only_41f.log", ARTIFACT / "clustered_early_only_41f_trace.csv", "none"),
    ("OneClusterEarlyLate_current_protocol", ARTIFACT / "one_cluster_full_early_41f.log", ARTIFACT / "one_cluster_full_early_41f_trace.csv", "identity_single_cluster"),
    ("FullSGCP_current_protocol", TABLE3 / "papg_41f.log", TABLE3 / "papg_41f_trace.csv", "inter_cluster_nms"),
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
        "grid_selection_mode": most_common([row.get("grid_selection_mode", "") for row in rows]),
        "grid_score_mode": most_common([row.get("grid_score_mode", "") for row in rows]),
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
    for label, log_path, trace_path, late_fusion in RUNS:
        log = parse_log(log_path)
        trace = parse_trace(trace_path, late_fusion)
        raw_mbps = (
            safe_int(log["payload_bytes"]) * 8.0
            / (safe_int(trace["unique_timestamps"]) * FRAME_INTERVAL_S)
            / 1e6
        )
        note = (
            "P12 current-protocol Table2 diagnostic; attentive; "
            "40MHz/10 target subchannels/60ms deadline; ns3 tb899 slot0.5 "
            "prb10 mcs28 symbols12."
        )
        if label == "FullSGCP_current_protocol":
            note += " Reuses Table3 PAPG current-protocol row."
        if label == "OneClusterEarlyLate_current_protocol":
            note += " Single-cluster late fusion is identity; reuses one-cluster early trace."
        if label == "PureLate_current_protocol":
            note += " Prediction-box sharing reference; no raw-LiDAR payload."
        rows.append({
            "label": label,
            "ap_03": log["ap_03"],
            "ap_05": log["ap_05"],
            "ap_07": log["ap_07"],
            "aggregate_ap_scope": "pooled evaluator over evaluated samples",
            "evaluated_samples": trace["unique_timestamps"],
            "trace_rows": log["trace_rows"],
            "unique_timestamps": trace["unique_timestamps"],
            "receiver_policy": trace["receiver_policy"],
            "late_fusion": late_fusion,
            "inter_cluster_late_fusion": "yes" if late_fusion == "inter_cluster_nms" else late_fusion,
            "fusion_method": "early",
            "resource_allocation": trace["resource_allocation"],
            "clustering": trace["clustering"],
            "upload_mode": trace["upload_mode"],
            "grid_selection_mode": trace["grid_selection_mode"],
            "grid_score_mode": trace["grid_score_mode"],
            "num_channels": trace["num_channels"],
            "bandwidth_mhz": trace["bandwidth_mhz"],
            "communication_deadline_ms": trace["communication_deadline_ms"],
            "channel_estimator": trace["channel_estimator"],
            "ns3_tb_size_bytes": trace["ns3_tb_size_bytes"],
            "ns3_slot_duration_ms": trace["ns3_slot_duration_ms"],
            "ns3_subchannel_prbs": trace["ns3_subchannel_prbs"],
            "ns3_symbols_per_slot": trace["ns3_symbols_per_slot"],
            "ns3_mcs": trace["ns3_mcs"],
            "cluster_count_mode": trace["cluster_count_mode"],
            "payload_bytes": log["payload_bytes"],
            "raw_lidar_mbps": "%.6f" % raw_mbps,
            "box_bytes": trace["box_bytes"],
            "box_mbps": "%.6f" % trace["box_mbps"],
            "total_mbps": "%.6f" % (raw_mbps + trace["box_mbps"]),
            "mbps": "%.6f" % (raw_mbps + trace["box_mbps"]),
            "avg_comm_bytes_per_trace_row": log["avg_comm_bytes_per_trace_row"],
            "avg_source_cavs": log["avg_source_cavs"],
            "avg_selected_grids": log["avg_selected_grids"],
            "checkpoint": "attentive",
            "box_sharing_mode": (
                "broadcast_boxes" if late_fusion in {"inter_cluster_nms", "prediction_nms"} else "none"
            ),
            "log_path": str(log_path.relative_to(REPO)),
            "trace_path": str(trace_path.relative_to(REPO)),
            "notes": note,
        })
    return rows


def write_csv(rows):
    outputs = [
        ARTIFACT / "table2_fusion_scaffold_current_protocol_20260722.csv",
        EXTERNAL_DATA / "table2_fusion_scaffold_current_protocol_20260722.csv",
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
    label_map = {
        "HeadOnly_current_protocol": "Head-only",
        "PureLate_current_protocol": "Pure late",
        "OneClusterEarlyOnly_current_protocol": "One-cluster early",
        "ClusteredEarlyOnly_current_protocol": "Clustered early",
        "OneClusterEarlyLate_current_protocol": "One-cluster early+late",
        "FullSGCP_current_protocol": "Full SGCP",
    }
    labels = [label_map.get(label, label) for label in df["label"]]
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(x - width, df["ap_03"], width, label="AP@0.3")
    ax.bar(x, df["ap_05"], width, label="AP@0.5")
    ax.bar(x + width, df["ap_07"], width, label="AP@0.7")
    for i, row in df.iterrows():
        ax.text(i, 0.025, "%.1f Mbps" % row["total_mbps"], ha="center",
                va="bottom", rotation=90, fontsize=7, color="0.25")
    ax.set_title("Current-protocol fusion diagnostic (40MHz/10ch/60ms)")
    ax.set_ylabel("Aggregate AP")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    plt.tight_layout()
    for suffix in ["png", "pdf"]:
        output = EXTERNAL_FIG / (
            "figure3_fusion_ablation_current_protocol_20260722.%s" % suffix
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
            "%s %s/%s/%s raw=%s box=%s total=%s" % (
                row["label"], row["ap_03"], row["ap_05"], row["ap_07"],
                row["raw_lidar_mbps"], row["box_mbps"], row["total_mbps"]
            )
        )


if __name__ == "__main__":
    main()
