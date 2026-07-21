# -*- coding: utf-8 -*-
"""Build current-protocol combined diagnostic Table A and Figures 1-2.

This artifact is intentionally diagnostic. It summarizes current 40MHz/10ch
/60ms rows already produced by component-table reruns, but it does not freeze
the paper-facing SGCP operating point because Table4 showed that N_max=2
dominates the strict default N_max=4 setting.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/tableA_current_protocol_20260722"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment")
DATA = EXPERIMENT / "data"
FIG = EXPERIMENT / "figures"

CURRENT_PROTOCOL_DEFAULTS = {
    "num_channels": "10",
    "bandwidth_mhz": "40",
    "communication_deadline_ms": "60",
    "channel_estimator": "ns3",
    "ns3_tb_size_bytes": "899",
    "ns3_slot_duration_ms": "0.5",
    "ns3_subchannel_prbs": "10",
    "ns3_symbols_per_slot": "12",
    "ns3_mcs": "28",
}


SOURCES = {
    "table2": DATA / "table2_fusion_scaffold_current_protocol_20260722.csv",
    "table3": DATA / "table3_scheduler_comparison_current_protocol_20260722.csv",
    "table4": DATA / "table4_parameter_sensitivity_current_protocol_20260722.csv",
    "table6": DATA / "table6_global_box_aggregation_current_protocol_20260722.csv",
    "sgcp_low": DATA / "sgcp_low_budget_40mhz_20260722.csv",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def find_row(rows, key, value):
    matches = [row for row in rows if row.get(key) == value]
    if not matches:
        raise RuntimeError("Missing row %s=%s" % (key, value))
    return matches[0]


def normalize_row(source, row, method, role, status, usage, **overrides):
    def protocol_value(key):
        value = row.get(key, "")
        if value in ("", None):
            return CURRENT_PROTOCOL_DEFAULTS.get(key, "")
        return value

    raw = {
        "method": method,
        "role": role,
        "source_table": source,
        "checkpoint": row.get("checkpoint", "attentive"),
        "late_fusion": row.get("late_fusion", ""),
        "clustering": row.get("clustering", ""),
        "resource_allocation": row.get("resource_allocation", ""),
        "receiver_policy": row.get("receiver_policy", ""),
        "num_channels": protocol_value("num_channels"),
        "bandwidth_mhz": protocol_value("bandwidth_mhz"),
        "communication_deadline_ms": protocol_value(
            "communication_deadline_ms"),
        "channel_estimator": protocol_value("channel_estimator"),
        "ns3_tb_size_bytes": protocol_value("ns3_tb_size_bytes"),
        "ns3_slot_duration_ms": protocol_value("ns3_slot_duration_ms"),
        "ns3_subchannel_prbs": protocol_value("ns3_subchannel_prbs"),
        "ns3_symbols_per_slot": protocol_value("ns3_symbols_per_slot"),
        "ns3_mcs": protocol_value("ns3_mcs"),
        "ap_03": row.get("ap_03", ""),
        "ap_05": row.get("ap_05", ""),
        "ap_07": row.get("ap_07", ""),
        "raw_lidar_mbps": row.get("raw_lidar_mbps", "0"),
        "box_mbps": row.get("box_mbps", "0"),
        "total_mbps": row.get("total_mbps", row.get("mbps", "")),
        "mbps": row.get("mbps", row.get("total_mbps", "")),
        "traffic_type": row.get("traffic_type", ""),
        "uses_sgcp_coalition": (
            "yes" if row.get("clustering") == "coalition_game" else "no"
        ),
        "uses_inter_cluster_late_fusion": (
            "yes" if row.get("late_fusion") in {
                "inter_cluster_nms", "prediction_nms", "global_box_nms"
            } else "no"
        ),
        "uses_raw_lidar_early_fusion": (
            "yes" if float(row.get("raw_lidar_mbps", "0") or 0) > 0 else "no"
        ),
        "box_sharing_mode": row.get("box_sharing_mode", ""),
        "trace_path": row.get("trace_path", ""),
        "baseline_status": status,
        "paper_usage": usage,
        "notes": row.get("notes", ""),
    }
    raw.update(overrides)
    return raw


def build_rows():
    table2 = read_rows(SOURCES["table2"])
    table3 = read_rows(SOURCES["table3"])
    table4 = read_rows(SOURCES["table4"])
    table6 = read_rows(SOURCES["table6"])
    sgcp_low = read_rows(SOURCES["sgcp_low"])

    rows = []
    rows.append(normalize_row(
        "Table2 current diagnostic",
        find_row(table2, "label", "HeadOnly_current_protocol"),
        "Head-only",
        "lower reference",
        "reference inside SGCP scaffold",
        "Diagnostic lower reference; not protocol-native baseline.",
    ))
    rows.append(normalize_row(
        "Table2 current diagnostic",
        find_row(table2, "label", "PureLate_current_protocol"),
        "Pure late",
        "prediction-sharing reference",
        "controlled prediction-sharing reference",
        "Shows strength of box-level sharing; must not be compared as raw-LiDAR scheduler.",
    ))
    rows.append(normalize_row(
        "Table6 current diagnostic",
        find_row(table6, "method", "FullPerception-PCS + global box current-protocol"),
        "FullPerception-PCS + global box",
        "normalized PCS scaffold with common box aggregation",
        "normalized scaffold; not original FullPerception protocol",
        "Diagnostic only; common box aggregation strongly boosts AP.",
    ))
    rows.append(normalize_row(
        "Table6 current diagnostic",
        find_row(table6, "method", "EdgeCooper V2V + global box current-protocol"),
        "EdgeCooper V2V + global box",
        "deadline-constrained EdgeCooper scaffold with common box aggregation",
        "normalized scaffold; not original EdgeCooper protocol",
        "Diagnostic only; common box aggregation strongly boosts AP.",
    ))
    rows.append(normalize_row(
        "Table3 current diagnostic",
        find_row(table3, "label", "RandomBudget_current_protocol"),
        "RandomBudget",
        "same-scaffold scheduler control",
        "SGCP-compatible scheduler diagnostic",
        "Diagnostic scheduler control under current protocol.",
    ))
    rows.append(normalize_row(
        "Table3 current diagnostic",
        find_row(table3, "label", "PACP_LiDAR_current_protocol"),
        "PACP-LiDAR",
        "same-scaffold adapted scheduler proxy",
        "SGCP-compatible scheduler proxy",
        "Diagnostic high-traffic scheduler proxy under current protocol.",
    ))
    rows.append(normalize_row(
        "Table3 current diagnostic",
        find_row(table3, "label", "SGCP-PAPG_current_protocol"),
        "SGCP-PAPG strict default",
        "strict default proposed scaffold",
        "proposed method diagnostic, unresolved operating point",
        "Not final paper row; Table4 shows N_max=2 dominates this strict default.",
    ))
    nmax2 = [
        row for row in table4
        if row.get("parameter") == "N_max" and row.get("setting") == "2"
    ][0]
    rows.append(normalize_row(
        "Table4 current diagnostic",
        nmax2,
        "SGCP-PAPG N_max=2 diagnostic",
        "candidate SGCP operating point from parameter sensitivity",
        "candidate proposed setting; requires full table reruns",
        "Promising diagnostic only; if adopted, Table2/3/5/TableA/Figure6 must be rerun around this operating point.",
    ))
    low = sgcp_low[0]
    rows.append(normalize_row(
        "40MHz addendum",
        low,
        "SGCP-PAPG low-budget cap4000",
        "low-budget proposed operating point",
        "proposed low-budget addendum",
        "Useful Pareto point with NS3 replay, but not a replacement for final main operating point.",
        raw_lidar_mbps="51.20",
        box_mbps="0.70",
        total_mbps=low.get("mbps", "51.90"),
        mbps=low.get("mbps", "51.90"),
        box_sharing_mode="broadcast_boxes",
    ))
    rows.append(normalize_row(
        "Table2 current diagnostic",
        find_row(table2, "label", "OneClusterEarlyOnly_current_protocol"),
        "Full 20-CAV early fusion",
        "full raw-sharing upper reference",
        "upper reference, not a baseline algorithm",
        "Use only as upper reference.",
        resource_allocation="full_sharing",
    ))
    return rows


def write_csv(rows):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    outputs = [
        ARTIFACT / "tableA_combined_current_protocol_diagnostic_20260722.csv",
        DATA / "tableA_combined_current_protocol_diagnostic_20260722.csv",
        DATA / "tableA_compact_current_protocol_diagnostic_20260722.csv",
    ]
    fieldnames = list(rows[0].keys())
    for output in outputs[:2]:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(output)
    compact_fields = [
        "method", "role", "checkpoint", "late_fusion", "clustering",
        "resource_allocation", "num_channels", "bandwidth_mhz",
        "communication_deadline_ms", "channel_estimator",
        "ns3_tb_size_bytes", "ns3_slot_duration_ms",
        "ns3_subchannel_prbs", "ns3_symbols_per_slot", "ns3_mcs",
        "ap_03", "ap_05", "ap_07", "mbps", "raw_lidar_mbps",
        "box_mbps", "total_mbps", "baseline_status", "paper_usage",
    ]
    with outputs[2].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=compact_fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in compact_fields} for row in rows])
    print(outputs[2])


def write_figures(rows):
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ["ap_03", "ap_05", "ap_07", "total_mbps"]:
        df[col] = pd.to_numeric(df[col])

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(df["total_mbps"], df["ap_05"], s=54)
    for _, row in df.iterrows():
        ax.annotate(
            row["method"],
            (row["total_mbps"], row["ap_05"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xlabel("Total Mbps (raw LiDAR + box)")
    ax.set_ylabel("Aggregate AP@0.5")
    ax.set_title("Current-protocol combined diagnostic Pareto")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        output = FIG / (
            "figure1_pareto_current_protocol_diagnostic_20260722.%s" % suffix
        )
        fig.savefig(output, dpi=220 if suffix == "png" else None)
        print(output)
    plt.close(fig)

    labels = [row["method"] for row in rows]
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - width, df["ap_03"], width, label="AP@0.3")
    ax.bar(x, df["ap_05"], width, label="AP@0.5")
    ax.bar(x + width, df["ap_07"], width, label="AP@0.7")
    for i, row in df.iterrows():
        ax.text(i, 0.025, "%.1f Mbps" % row["total_mbps"], ha="center",
                va="bottom", rotation=90, fontsize=7, color="0.25")
    ax.set_title("Current-protocol combined AP diagnostic")
    ax.set_ylabel("Aggregate AP")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        output = FIG / (
            "figure2_combined_current_protocol_diagnostic_20260722.%s" % suffix
        )
        fig.savefig(output, dpi=220 if suffix == "png" else None)
        print(output)
    plt.close(fig)


def main():
    rows = build_rows()
    write_csv(rows)
    write_figures(rows)
    for row in rows:
        print("%s %s/%s/%s total=%s" % (
            row["method"], row["ap_03"], row["ap_05"], row["ap_07"],
            row["total_mbps"]))


if __name__ == "__main__":
    main()
