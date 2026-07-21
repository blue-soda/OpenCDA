# -*- coding: utf-8 -*-
"""Run and build current-protocol SGCP parameter-sensitivity diagnostics.

This script intentionally treats the channel-count sweep as a resource-stress
diagnostic: the formal protocol fixes 40 MHz / 10 target subchannels, while
the channel sweep varies the target subchannel count to test budget response.
"""

import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table4_current_protocol_20260722"
BASE_ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table3_current_protocol_20260722"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment")
EXTERNAL_DATA = EXPERIMENT / "data"
EXTERNAL_FIG = EXPERIMENT / "figures"

BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1

BASE_ARGS = [
    "-m", "opencda.tools.offline_inference",
    "--dataset-root", r"D:\Data\Carla",
    "--scenario-id", "2026_07_15_01_26_56",
    "--ego-cav-id", "1",
    "--max-frames", "0",
    "--sgcp-constrained",
    "--clustering", "coalition_game",
    "--sgcp-receiver-policy", "all-cluster-heads",
    "--sgcp-upload-mode", "grid",
    "--resource-allocation", "perception_aware_potential_game",
    "--sgcp-inter-cluster-late-fusion",
    "--rho-th", "3",
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

RUNS = [
    {
        "label": "rho_th=1",
        "group": "rho_th",
        "setting": "1",
        "stem": "rho1",
        "extra": ["--rho-th", "1"],
        "note": "current-protocol rho sweep",
    },
    {
        "label": "rho_th=2",
        "group": "rho_th",
        "setting": "2",
        "stem": "rho2",
        "extra": ["--rho-th", "2"],
        "note": "current-protocol rho sweep",
    },
    {
        "label": "rho_th=3",
        "group": "rho_th",
        "setting": "3",
        "stem": "base",
        "reuse_base": True,
        "note": "current-protocol base point reused from Table3 diagnostic",
    },
    {
        "label": "N_max=2",
        "group": "N_max",
        "setting": "2",
        "stem": "nmax2",
        "extra": ["--n-max", "2"],
        "note": "current-protocol cluster-capacity sweep",
    },
    {
        "label": "N_max=3",
        "group": "N_max",
        "setting": "3",
        "stem": "nmax3",
        "extra": ["--n-max", "3"],
        "note": "current-protocol cluster-capacity sweep",
    },
    {
        "label": "N_max=4",
        "group": "N_max",
        "setting": "4",
        "stem": "base",
        "reuse_base": True,
        "note": "current-protocol base point reused from Table3 diagnostic",
    },
    {
        "label": "N_max=5",
        "group": "N_max",
        "setting": "5",
        "stem": "nmax5",
        "extra": ["--n-max", "5"],
        "note": "current-protocol cluster-capacity sweep",
    },
    {
        "label": "N_max=6",
        "group": "N_max",
        "setting": "6",
        "stem": "nmax6",
        "extra": ["--n-max", "6"],
        "note": "current-protocol cluster-capacity sweep",
    },
    {
        "label": "channels=5",
        "group": "target_subchannels",
        "setting": "5",
        "stem": "ch5",
        "extra": ["--num-channels", "5"],
        "note": "resource-stress diagnostic: target subchannel count changed",
    },
    {
        "label": "channels=10",
        "group": "target_subchannels",
        "setting": "10",
        "stem": "base",
        "reuse_base": True,
        "note": "current-protocol base point reused from Table3 diagnostic",
    },
    {
        "label": "channels=20",
        "group": "target_subchannels",
        "setting": "20",
        "stem": "ch20",
        "extra": ["--num-channels", "20"],
        "note": "resource-stress diagnostic: target subchannel count changed",
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


def _replace_arg(args, name, value):
    updated = list(args)
    if name in updated:
        idx = updated.index(name)
        updated[idx + 1] = str(value)
    else:
        updated.extend([name, str(value)])
    return updated


def _safe_int(value):
    try:
        if value in ("", None):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _most_common(values):
    values = [value for value in values if value not in ("", None)]
    if not values:
        return ""
    return max(set(values), key=values.count)


def command_for(run):
    args = list(BASE_ARGS)
    extra = run.get("extra", [])
    for idx in range(0, len(extra), 2):
        args = _replace_arg(args, extra[idx], extra[idx + 1])
    trace_path = ARTIFACT / ("%s_trace.csv" % run["stem"])
    args.extend(["--sgcp-trace-output", str(trace_path)])
    return [sys.executable] + args


def ensure_base_artifact():
    base_log = ARTIFACT / "base.log"
    base_trace = ARTIFACT / "base_trace.csv"
    if not base_log.exists():
        shutil.copyfile(BASE_ARTIFACT / "papg_41f.log", base_log)
    if not base_trace.exists():
        shutil.copyfile(BASE_ARTIFACT / "papg_41f_trace.csv", base_trace)


def run_experiments():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    ensure_base_artifact()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for run in RUNS:
        if run.get("reuse_base"):
            continue
        log_path = ARTIFACT / ("%s.log" % run["stem"])
        trace_path = ARTIFACT / ("%s_trace.csv" % run["stem"])
        if log_path.exists() and trace_path.exists():
            print("skip existing %s" % run["stem"])
            continue
        cmd = command_for(run)
        print("running %s" % run["label"])
        print(" ".join(cmd))
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            proc = subprocess.run(
                cmd, cwd=str(REPO), env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True
            )
        if proc.returncode != 0:
            raise RuntimeError("%s failed with exit code %s" % (run["label"], proc.returncode))


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


def parse_trace(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Empty trace: %s" % path)
    timestamps = sorted({row.get("timestamp", "") for row in rows if row.get("timestamp", "")})
    by_sample = {}
    for row in rows:
        timestamp = row.get("timestamp", "")
        receiver = row.get("receiver_id", "")
        if not timestamp or not receiver:
            continue
        key = (timestamp, receiver)
        by_sample[key] = max(by_sample.get(key, 0), _safe_int(row.get("pred_boxes")))
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values()
        if boxes > 0
    )
    duration_s = max(len(timestamps) * FRAME_INTERVAL_S, 1e-9)
    return {
        "unique_timestamps": str(len(timestamps)),
        "receiver_policy": _most_common([row.get("receiver_policy", "") for row in rows]),
        "late_fusion": "inter_cluster_nms",
        "clustering": _most_common([row.get("clustering", "") for row in rows]),
        "resource_allocation": _most_common([row.get("resource_allocation", "") for row in rows]),
        "upload_mode": _most_common([row.get("upload_mode", "") for row in rows]),
        "grid_selection_mode": _most_common([row.get("grid_selection_mode", "") for row in rows]),
        "grid_score_mode": _most_common([row.get("grid_score_mode", "") for row in rows]),
        "num_channels": _most_common([row.get("num_channels", "") for row in rows]),
        "bandwidth_mhz": _most_common([row.get("bandwidth_mhz", "") for row in rows]),
        "communication_deadline_ms": _most_common(
            [row.get("communication_deadline_ms", "") for row in rows]
        ),
        "channel_estimator": _most_common([row.get("channel_estimator", "") for row in rows]),
        "ns3_tb_size_bytes": _most_common([row.get("ns3_tb_size_bytes", "") for row in rows]),
        "ns3_slot_duration_ms": _most_common([row.get("ns3_slot_duration_ms", "") for row in rows]),
        "ns3_subchannel_prbs": _most_common([row.get("ns3_subchannel_prbs", "") for row in rows]),
        "ns3_symbols_per_slot": _most_common([row.get("ns3_symbols_per_slot", "") for row in rows]),
        "ns3_mcs": _most_common([row.get("ns3_mcs", "") for row in rows]),
        "cluster_count_mode": _most_common([row.get("cluster_count", "") for row in rows]),
        "box_bytes": str(box_bytes),
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
    }


def build_rows():
    rows = []
    for run in RUNS:
        log = parse_log(ARTIFACT / ("%s.log" % run["stem"]))
        trace = parse_trace(ARTIFACT / ("%s_trace.csv" % run["stem"]))
        duration_s = _safe_int(trace["unique_timestamps"]) * FRAME_INTERVAL_S
        raw_mbps = _safe_int(log["payload_bytes"]) * 8.0 / max(duration_s, 1e-9) / 1e6
        total_mbps = raw_mbps + trace["box_mbps"]
        note = (
            "%s; attentive; 40MHz total bandwidth; 60ms communication deadline; "
            "NS3 estimator tb899/slot0.5/prb10/mcs28/symbols12; "
            "box communication included when inter-cluster NMS is used."
        ) % run["note"]
        if run["group"] == "target_subchannels":
            note += (
                " This row intentionally varies target subchannel count and is "
                "a resource-sensitivity diagnostic, not the fixed-protocol main point."
            )
        rows.append({
            "parameter": run["group"],
            "setting": run["setting"],
            "label": run["label"],
            "ap_03": log["ap_03"],
            "ap_05": log["ap_05"],
            "ap_07": log["ap_07"],
            "aggregate_ap_scope": "pooled evaluator over evaluated samples",
            "evaluated_samples": trace["unique_timestamps"],
            "trace_rows": log["trace_rows"],
            "unique_timestamps": trace["unique_timestamps"],
            "receiver_policy": trace["receiver_policy"],
            "late_fusion": trace["late_fusion"],
            "inter_cluster_late_fusion": "yes",
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
            "total_mbps": "%.6f" % total_mbps,
            "mbps": "%.6f" % total_mbps,
            "avg_comm_bytes_per_trace_row": log["avg_comm_bytes_per_trace_row"],
            "avg_source_cavs": log["avg_source_cavs"],
            "avg_selected_grids": log["avg_selected_grids"],
            "checkpoint": "attentive",
            "box_sharing_mode": "broadcast_boxes",
            "log_path": str((ARTIFACT / ("%s.log" % run["stem"])).relative_to(REPO)),
            "trace_path": str((ARTIFACT / ("%s_trace.csv" % run["stem"])).relative_to(REPO)),
            "notes": note,
        })
    return rows


def write_csv(rows):
    fieldnames = list(rows[0].keys())
    outputs = [
        ARTIFACT / "table4_parameter_sensitivity_current_protocol_20260722.csv",
        EXTERNAL_DATA / "table4_parameter_sensitivity_current_protocol_20260722.csv",
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(output)


def write_figure(rows):
    EXTERNAL_FIG.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ["setting", "ap_03", "ap_05", "ap_07", "total_mbps"]:
        df[col] = pd.to_numeric(df[col])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    groups = [
        ("rho_th", "rho_th sweep"),
        ("N_max", "cluster capacity"),
        ("target_subchannels", "target subchannels"),
    ]
    for ax, (group, title) in zip(axes, groups):
        part = df[df["parameter"] == group].sort_values("setting")
        x = np.arange(len(part))
        width = 0.22
        ax.bar(x - width, part["ap_03"], width, label="AP@0.3")
        ax.bar(x, part["ap_05"], width, label="AP@0.5")
        ax.bar(x + width, part["ap_07"], width, label="AP@0.7")
        for i, (_, row) in enumerate(part.iterrows()):
            ax.text(i, 0.03, "%.1f Mbps" % row["total_mbps"],
                    ha="center", va="bottom", rotation=90, fontsize=7, color="0.25")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in part["setting"]])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Aggregate AP")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Current-protocol SGCP parameter sensitivity diagnostics", y=1.02)
    plt.tight_layout()
    for suffix in ["png", "pdf"]:
        output = EXTERNAL_FIG / (
            "figure5_parameter_sensitivity_current_protocol_20260722.%s" % suffix
        )
        fig.savefig(output, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        print(output)
    plt.close(fig)


def main():
    run_experiments()
    rows = build_rows()
    write_csv(rows)
    write_figure(rows)
    for row in rows:
        print(
            "%s %s/%s/%s total=%s" % (
                row["label"], row["ap_03"], row["ap_05"], row["ap_07"], row["total_mbps"]
            )
        )


if __name__ == "__main__":
    main()
