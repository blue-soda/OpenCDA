# -*- coding: utf-8 -*-
"""Run and build current-protocol Table 6 global-box diagnostics."""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table6_current_protocol_20260722"
TABLE2 = REPO / "docs/doc_workspace/SGCP/artifacts/table2_current_protocol_20260722"
TABLE3 = REPO / "docs/doc_workspace/SGCP/artifacts/table3_current_protocol_20260722"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment")
EXTERNAL_DATA = EXPERIMENT / "data"
EXTERNAL_FIG = EXPERIMENT / "figures"

BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1

COMMON_ARGS = [
    "-m", "opencda.tools.offline_inference",
    "--dataset-root", r"D:\Data\Carla",
    "--scenario-id", "2026_07_15_01_26_56",
    "--ego-cav-id", "1",
    "--max-frames", "0",
    "--num-channels", "10",
    "--bandwidth-mhz", "40",
    "--communication-deadline-ms", "60",
    "--channel-estimator", "ns3",
    "--ns3-tb-size-bytes", "899",
    "--ns3-slot-duration-ms", "0.5",
    "--ns3-subchannel-prbs", "10",
    "--ns3-symbols-per-slot", "12",
    "--ns3-mcs", "28",
]

RUN_COMMANDS = [
    {
        "label": "FullPerception-PCS + global box",
        "stem": "pcs_global_box_41f",
        "args": COMMON_ARGS + [
            "--sgcp-constrained",
            "--resource-allocation", "fullperception_pcs",
            "--sgcp-receiver-policy", "all-cavs",
            "--sgcp-inter-cluster-late-fusion",
            "--clustering", "singleton",
        ],
    },
    {
        "label": "EdgeCooper V2V + global box",
        "stem": "edgecooper_global_box_41f",
        "args": COMMON_ARGS + [
            "--selective-sharing-baseline", "edgecooper_global",
            "--selective-member-budget", "3",
            "--selective-grid-budget", "117",
            "--selective-frame-deadline-ms", "60",
            "--edgecooper-global-comm-range-m", "35",
            "--sgcp-receiver-policy", "all-cavs",
            "--sgcp-inter-cluster-late-fusion",
            "--clustering", "singleton",
        ],
    },
]

ROWS = [
    {
        "method": "Pure late current-protocol",
        "role": "local prediction + common global box aggregation reference",
        "late_fusion": "prediction_nms",
        "log": TABLE2 / "pure_late_41f.log",
        "trace": TABLE2 / "pure_late_41f_trace.csv",
        "resource_allocation_override": "local_detection",
        "receiver_policy_override": "all-cavs",
        "notes": "Prediction-box sharing reference; no raw-LiDAR payload.",
    },
    {
        "method": "FullPerception-PCS + global box current-protocol",
        "role": "PCS raw-LiDAR adaptation with common scene-level box aggregation",
        "late_fusion": "global_box_nms",
        "log": ARTIFACT / "pcs_global_box_41f.log",
        "trace": ARTIFACT / "pcs_global_box_41f_trace.csv",
        "notes": (
            "All 20 CAVs are potential receivers under singleton clustering; "
            "the PCS schedule is evaluated with common scene-level box NMS."
        ),
    },
    {
        "method": "EdgeCooper V2V + global box current-protocol",
        "role": "deadline-constrained EdgeCooper V2V adaptation with common scene-level box aggregation",
        "late_fusion": "global_box_nms",
        "log": ARTIFACT / "edgecooper_global_box_41f.log",
        "trace": ARTIFACT / "edgecooper_global_box_41f_trace.csv",
        "notes": (
            "Uses singleton receiver universe, original EdgeCooper-style "
            "greedy endpoint-disjoint matching, and one shared 60ms frame budget."
        ),
    },
    {
        "method": "SGCP-PAPG strict current-protocol",
        "role": "proposed clustered early fusion + inter-cluster box aggregation",
        "late_fusion": "inter_cluster_nms",
        "log": TABLE3 / "papg_41f.log",
        "trace": TABLE3 / "papg_41f_trace.csv",
        "notes": (
            "Strict default PAPG current-protocol diagnostic; not the final "
            "paper operating point until N_max/budget decision is resolved."
        ),
    },
    {
        "method": "Full 20-CAV early fusion current-protocol",
        "role": "full raw-sharing upper reference",
        "late_fusion": "none",
        "log": TABLE2 / "one_cluster_full_early_41f.log",
        "trace": TABLE2 / "one_cluster_full_early_41f_trace.csv",
        "resource_allocation_override": "full_sharing",
        "notes": "Full raw-sharing upper reference; not a baseline algorithm.",
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


def run_experiments():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for run in RUN_COMMANDS:
        log_path = ARTIFACT / ("%s.log" % run["stem"])
        trace_path = ARTIFACT / ("%s_trace.csv" % run["stem"])
        if log_path.exists() and trace_path.exists():
            print("skip existing %s" % run["stem"])
            continue
        cmd = [sys.executable] + run["args"] + [
            "--sgcp-trace-output", str(trace_path)
        ]
        print("running %s" % run["label"])
        print(" ".join(cmd))
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            proc = subprocess.run(
                cmd, cwd=str(REPO), env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError("%s failed with exit code %s" % (
                run["label"], proc.returncode))


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
    if late_fusion in {"inter_cluster_nms", "prediction_nms", "global_box_nms"}:
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
        "selective_frame_deadline_ms": most_common([row.get("selective_frame_deadline_ms", "") for row in rows]),
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
    for item in ROWS:
        log = parse_log(item["log"])
        trace = parse_trace(item["trace"], item["late_fusion"])
        duration_s = safe_int(trace["unique_timestamps"]) * FRAME_INTERVAL_S
        raw_mbps = safe_int(log["payload_bytes"]) * 8.0 / max(duration_s, 1e-9) / 1e6
        total_mbps = raw_mbps + trace["box_mbps"]
        rows.append({
            "method": item["method"],
            "role": item["role"],
            "checkpoint": "attentive",
            "late_fusion": item["late_fusion"],
            "clustering": trace["clustering"],
            "resource_allocation": item.get(
                "resource_allocation_override",
                trace["resource_allocation"]),
            "receiver_policy": item.get(
                "receiver_policy_override",
                trace["receiver_policy"]),
            "num_channels": trace["num_channels"],
            "bandwidth_mhz": trace["bandwidth_mhz"],
            "communication_deadline_ms": trace["communication_deadline_ms"],
            "selective_frame_deadline_ms": trace["selective_frame_deadline_ms"],
            "channel_estimator": trace["channel_estimator"],
            "ns3_tb_size_bytes": trace["ns3_tb_size_bytes"],
            "ns3_slot_duration_ms": trace["ns3_slot_duration_ms"],
            "ns3_subchannel_prbs": trace["ns3_subchannel_prbs"],
            "ns3_symbols_per_slot": trace["ns3_symbols_per_slot"],
            "ns3_mcs": trace["ns3_mcs"],
            "ap_03": log["ap_03"],
            "ap_05": log["ap_05"],
            "ap_07": log["ap_07"],
            "aggregate_ap_scope": "pooled evaluator over evaluated samples",
            "evaluated_samples": trace["unique_timestamps"],
            "trace_rows": log["trace_rows"],
            "receiver_samples_per_frame": trace["cluster_count_mode"],
            "traffic_type": (
                "box/prediction overhead"
                if raw_mbps == 0 else
                "raw LiDAR + box overhead"
                if trace["box_mbps"] > 0 else
                "raw LiDAR"
            ),
            "uses_sgcp_clustering": "yes" if trace["clustering"] == "coalition_game" else "no",
            "uses_inter_cluster_late_fusion": (
                "yes" if item["late_fusion"] in {
                    "inter_cluster_nms", "prediction_nms", "global_box_nms"
                } else "no"
            ),
            "uses_raw_lidar_early_fusion": "yes" if raw_mbps > 0 else "no",
            "payload_bytes": log["payload_bytes"],
            "raw_lidar_mbps": "%.6f" % raw_mbps,
            "box_bytes": trace["box_bytes"],
            "box_mbps": "%.6f" % trace["box_mbps"],
            "total_mbps": "%.6f" % total_mbps,
            "mbps": "%.6f" % total_mbps,
            "box_sharing_mode": (
                "broadcast_boxes"
                if item["late_fusion"] in {
                    "inter_cluster_nms", "prediction_nms", "global_box_nms"
                } else "none"
            ),
            "avg_comm_bytes_per_trace_row": log["avg_comm_bytes_per_trace_row"],
            "avg_source_cavs": log["avg_source_cavs"],
            "avg_selected_grids": log["avg_selected_grids"],
            "log_path": str(item["log"].relative_to(REPO)),
            "trace_path": str(item["trace"].relative_to(REPO)),
            "notes": (
                item["notes"] + " Current-protocol diagnostic; attentive; "
                "40MHz/10 target subchannels/60ms deadline; ns3 tb899 slot0.5 "
                "prb10 mcs28 symbols12."
            ),
        })
    return rows


def write_csv(rows):
    outputs = [
        ARTIFACT / "table6_global_box_aggregation_current_protocol_20260722.csv",
        EXTERNAL_DATA / "table6_global_box_aggregation_current_protocol_20260722.csv",
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
        "Pure late",
        "PCS+box",
        "EdgeCooper+box",
        "SGCP",
        "Full20 early",
    ]
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - width, df["ap_03"], width, label="AP@0.3")
    ax.bar(x, df["ap_05"], width, label="AP@0.5")
    ax.bar(x + width, df["ap_07"], width, label="AP@0.7")
    for i, row in df.iterrows():
        ax.text(i, 0.025, "%.1f Mbps" % row["total_mbps"], ha="center",
                va="bottom", rotation=90, fontsize=7, color="0.25")
    ax.set_title("Current-protocol global box aggregation diagnostic")
    ax.set_ylabel("Aggregate AP")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    plt.tight_layout()
    for suffix in ["png", "pdf"]:
        output = EXTERNAL_FIG / (
            "figure8_global_box_aggregation_current_protocol_20260722.%s" % suffix
        )
        fig.savefig(output, dpi=220 if suffix == "png" else None)
        print(output)
    plt.close(fig)


def main():
    run_experiments()
    rows = build_rows()
    write_csv(rows)
    write_figure(rows)
    for row in rows:
        print("%s %s/%s/%s total=%s" % (
            row["method"], row["ap_03"], row["ap_05"], row["ap_07"],
            row["total_mbps"]))


if __name__ == "__main__":
    main()
