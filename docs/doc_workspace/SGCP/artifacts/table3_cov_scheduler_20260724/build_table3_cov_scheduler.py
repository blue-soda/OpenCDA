# -*- coding: utf-8 -*-
"""Build Table 3 from formal SGCP-CV scaffold reruns."""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/table3_cov_scheduler_20260724"
EXPERIMENT = Path(r"C:\Workspace\2026-7-papers\infocom\SGCP\experiment")
COMPUTE = REPO / "docs/doc_workspace/SGCP/artifacts/compute_profile_20260722"
SGCP_BASE = REPO / "docs/doc_workspace/SGCP/artifacts/parameter_sensitivity_cv_20260723"

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
        "label": "SGCP-CV",
        "display": "SGCP-CV",
        "scheduler": "cov_potential_game",
        "log": SGCP_BASE / "base.log",
        "trace": SGCP_BASE / "base_trace.csv",
        "note_row": False,
    },
    {
        "label": "Cluster-head late only",
        "display": "Cluster-head late only",
        "scheduler": "local_detection_head_only",
        "log": ARTIFACT / "head_only_k2.log",
        "trace": ARTIFACT / "head_only_k2_trace.csv",
        "note_row": False,
    },
    {
        "label": "FullPerception-PCS",
        "display": "FullPerception-PCS",
        "scheduler": "fullperception_pcs",
        "log": ARTIFACT / "pcs_k2.log",
        "trace": ARTIFACT / "pcs_k2_trace.csv",
        "note_row": False,
    },
    {
        "label": "Random budget",
        "display": "Random budget",
        "scheduler": "selective_random",
        "log": ARTIFACT / "random_k2.log",
        "trace": ARTIFACT / "random_k2_trace.csv",
        "note_row": False,
    },
    {
        "label": "Density greedy",
        "display": "Density greedy",
        "scheduler": "selective_density",
        "log": ARTIFACT / "density_k2.log",
        "trace": ARTIFACT / "density_k2_trace.csv",
        "note_row": False,
    },
    {
        "label": "Link-aware density",
        "display": "Link-aware density",
        "scheduler": "selective_communication_aware",
        "log": ARTIFACT / "linkaware_k2.log",
        "trace": ARTIFACT / "linkaware_k2_trace.csv",
        "note_row": False,
    },
    {
        "label": "PACP-LiDAR",
        "display": "PACP-LiDAR",
        "scheduler": "selective_pacp_lidar",
        "log": ARTIFACT / "pacp_lidar_k2.log",
        "trace": ARTIFACT / "pacp_lidar_k2_trace.csv",
        "note_row": False,
    },
    {
        "label": "EdgeCooper-HD",
        "display": "EdgeCooper-HD",
        "scheduler": "selective_edgecooper_global_hd",
        "log": ARTIFACT / "edgecooper_hd_k2.log",
        "trace": ARTIFACT / "edgecooper_hd_k2_trace.csv",
        "note_row": False,
    },
    {
        "label": "FullPerception-PCS K1 note",
        "display": "FullPerception-PCS",
        "scheduler": "fullperception_pcs",
        "log": ARTIFACT / "pcs_k1_note.log",
        "trace": ARTIFACT / "pcs_k1_note_trace.csv",
        "note_row": True,
    },
    {
        "label": "EdgeCooper-HD K1 note",
        "display": "EdgeCooper-HD",
        "scheduler": "selective_edgecooper_global_hd",
        "log": ARTIFACT / "edgecooper_hd_k1_note.log",
        "trace": ARTIFACT / "edgecooper_hd_k1_note_trace.csv",
        "note_row": True,
    },
]


def safe_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
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
        "avg_comm_bytes": float(avg_comm),
        "payload_bytes": int(total_comm),
        "avg_source_cavs": float(avg_sources),
        "avg_selected_grids": float(avg_grids),
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
        key = (row.get("timestamp", ""), row.get("receiver_id", ""))
        if not key[0] or not key[1]:
            continue
        by_sample[key] = max(by_sample.get(key, 0),
                             safe_int(row.get("pred_boxes")))
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values()
        if boxes > 0
    )
    duration_s = max(len(timestamps) * FRAME_INTERVAL_S, 1e-9)
    return {
        "rows": rows,
        "frames": len(timestamps),
        "box_bytes": box_bytes,
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
        "receiver_policy": most_common(row.get("receiver_policy", "")
                                       for row in rows),
        "clustering": most_common(row.get("clustering", "") for row in rows),
        "max_senders_per_receiver": most_common(
            row.get("max_senders_per_receiver", "") for row in rows),
        "avg_frame_comm_time_ms": sum(
            safe_float(row.get("frame_comm_time_ms")) for row in rows
        ) / max(len(rows), 1),
    }


def extract_json_from_out(src, dst):
    text = src.read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("No JSON object in %s" % src)
    data = json.loads(text[start:end + 1])
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dst


def build_metric_rows():
    rows = []
    for run in RUNS:
        log = parse_log(run["log"])
        trace = parse_trace(run["trace"])
        raw_mbps = (
            log["payload_bytes"] * 8.0 /
            max(trace["frames"] * FRAME_INTERVAL_S, 1e-9) / 1e6
        )
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
            "avg_frame_comm_time_ms": "%.2f" % trace["avg_frame_comm_time_ms"],
            "max_senders_per_receiver": trace["max_senders_per_receiver"],
            "trace_path": str(run["trace"]),
            "note_row": "yes" if run["note_row"] else "no",
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_compute_profile(metric_csv):
    singleton = extract_json_from_out(
        COMPUTE / "attentive_singleton_forward_flops.out",
        ARTIFACT / "attentive_singleton_forward_flops.json")
    dense = extract_json_from_out(
        COMPUTE / "attentive_full20_forward_flops.out",
        ARTIFACT / "attentive_full20_forward_flops.json")
    output_csv = ARTIFACT / "table3_cov_scheduler_compute_20260724.csv"
    output_md = ARTIFACT / "table3_cov_scheduler_compute_20260724.md"
    cmd = [
        sys.executable,
        "-m", "opencda.tools.sgcp_compute_profile",
        "--metrics-csv", str(metric_csv),
        "--calibration-json", str(singleton),
        "--dense-calibration-json", str(dense),
        "--output-csv", str(output_csv),
        "--summary-md", str(output_md),
    ]
    for row in RUNS:
        cmd.extend(["--method", "%s=%s" % (row["label"], row["trace"])])
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


def build_markdown(rows, compute_rows):
    main_rows = [row for row in rows if row["note_row"] == "no"]
    k1_rows = [row for row in rows if row["note_row"] == "yes"]
    lines = [
        "# Table 3. SGCP-compatible Scheduler Comparison",
        "",
        "Protocol: attentive detector, v2xp_cluster_carla 41-frame offline replay, 20 CAVs, 40 MHz / 10 target subchannels, NS3-calibrated channel estimator (`tb_size=899`, `symbols=12`, `mcs=28`), `cov_coalition_game` clustering, all cluster heads as receivers, grid raw-LiDAR upload, inter-cluster box NMS. SGCP-CV uses the selected main operating point from the Raw LiDAR Mbps Budget sweep (`200 Mbps` row). Baseline scheduler rows report K=2 receiver-side concurrent inbound links where applicable, aligning their receiver capability with SGCP.",
        "",
        "| Method | Late fusion | Clustering | Scheduler/protocol | Receiver policy | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg grids |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        comp = compute_rows[row["label"]]
        lines.append(
            "| {display} | {late_fusion} | {clustering} | {scheduler} | "
            "{receiver_policy} | {ap_03} | {ap_05} | {ap_07} | "
            "{raw} | {box} | {total} | {gflops} | {grids} |".format(
                display=row["display"],
                late_fusion=row["late_fusion"],
                clustering=row["clustering"],
                scheduler=row["scheduler"],
                receiver_policy=row["receiver_policy"],
                ap_03=row["ap_03"],
                ap_05=row["ap_05"],
                ap_07=row["ap_07"],
                raw=fmt(row["raw_lidar_mbps"]),
                box=fmt(row["box_mbps"]),
                total=fmt(row["total_mbps"]),
                gflops=fmt(comp["input_adjusted_detector_gflops_per_frame"]),
                grids=fmt(row["avg_selected_grids"]),
            )
        )
    k1_note = []
    for row in k1_rows:
        comp = compute_rows[row["label"]]
        k1_note.append(
            "{display} K=1: AP {ap03}/{ap05}/{ap07}, Total {total} Mbps, "
            "GFLOPs/frame {gflops}".format(
                display=row["display"],
                ap03=row["ap_03"],
                ap05=row["ap_05"],
                ap07=row["ap_07"],
                total=fmt(row["total_mbps"]),
                gflops=fmt(comp["input_adjusted_detector_gflops_per_frame"]),
            )
        )
    lines.extend([
        "",
        "Notes:",
        "- This is a scheduler comparison under the same SGCP-compatible clustering and late-aggregation scaffold, not a protocol-native baseline table.",
        "- `Raw Mbps` is scheduled raw-LiDAR payload; `Box Mbps` is inter-cluster detection-box aggregation payload; `Total Mbps = Raw Mbps + Box Mbps`.",
        "- K=2 is used in the main table to align baseline receiver capability with SGCP. " + "; ".join(k1_note) + ".",
        "- Admission-budget parameters are internal scheduler controls and are not reported as paper-facing table columns; feasibility is judged from the resulting payload and NS3-calibrated delay diagnostics.",
        "",
    ])
    return "\n".join(lines)


def update_external_docs():
    protocol_path = EXPERIMENT / "00_protocol_and_metrics.md"
    text = protocol_path.read_text(encoding="utf-8")
    text = re.sub(
        r"\| Scheduler admission budget \|[^\n]+\n",
        "",
        text,
    )
    text = text.replace(
        "| Evaluated samples | Number of receiver-frame samples evaluated by the pooled AP evaluator. |\n",
        "",
    )
    if "Admission-budget parameters are internal scheduler controls" not in text:
        text = text.replace(
            "All tables keep AP@0.7 in the clean package. The paper-writing agent can decide whether to show only AP@0.3/AP@0.5 in the manuscript.\n",
            "All tables keep AP@0.7 in the clean package. The paper-writing agent can decide whether to show only AP@0.3/AP@0.5 in the manuscript.\n\nAdmission-budget parameters are internal scheduler controls and are not reported as paper-facing table columns. Reported communication cost is payload Mbps; latency feasibility should be checked with NS3-calibrated delay diagnostics.\n",
        )
    protocol_path.write_text(text, encoding="utf-8")

    table1_path = EXPERIMENT / "01_protocol_native_baselines.md"
    text = table1_path.read_text(encoding="utf-8")
    text = text.replace(
        "| Method | Late fusion | Clustering | Scheduler/protocol | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Samples | Scope |\n",
        "| Method | Late fusion | Clustering | Scheduler/protocol | AP@0.3 | AP@0.5 | AP@0.7 | Raw Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Scope |\n",
    )
    text = text.replace(
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n",
    )
    new_lines = []
    for line in text.splitlines():
        if line.startswith("|") and line.count("|") >= 13 and not line.startswith("| Method") and not line.startswith("|---"):
            parts = line.split("|")
            # Drop the Samples column between GFLOPs/frame and Scope.
            if len(parts) == 15:
                parts = parts[:12] + parts[13:]
                line = "|".join(parts)
        new_lines.append(line)
    table1_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    plot_path = EXPERIMENT / "07_plot_data_and_suggestions.md"
    text = plot_path.read_text(encoding="utf-8")
    text = text.replace(
        "; rerun non-SGCP rows under `cov_coalition_game` before final paper use",
        "",
    )
    plot_path.write_text(text, encoding="utf-8")

    quality_path = EXPERIMENT / "08_data_quality_and_remaining_work.md"
    text = quality_path.read_text(encoding="utf-8")
    text = text.replace(
        "| Scheduler/scaffold rows fully rerun under `cov_coalition_game` | Table 3 and older heuristic rows in Table 4 still include retained scaffold baselines from before the formal C/V cleanup. | Rerun these rows with `cov_coalition_game` / `cov_potential_game` if they will be used as final paper-facing comparisons. |",
        "| Scheduler/scaffold rows fully rerun under `cov_coalition_game` | Table 3 has been rerun under the formal C/V scaffold. Table 4 heuristic clustering rows are intentionally left for the next baseline-selection discussion. | Decide Table 4 clustering baselines, then rerun only the selected algorithms. |",
    )
    quality_path.write_text(text, encoding="utf-8")

    params_path = EXPERIMENT / "05_baseline_reproduction_parameters.md"
    text = params_path.read_text(encoding="utf-8")
    text = text.replace(
        "| SGCP-compatible clustering | `cov_coalition_game` for final SGCP-CV rows; older retained scaffold rows may still use `coalition_game` |",
        "| SGCP-compatible clustering | `cov_coalition_game` for Table 3 and final SGCP-CV rows; Table 4 heuristic clustering baselines remain pending selection/rerun |",
    )
    params_path.write_text(text, encoding="utf-8")


def write_manifest():
    lines = [
        "# Manifest",
        "",
        "| File | Purpose |",
        "|---|---|",
    ]
    for path in sorted(EXPERIMENT.glob("*.md")):
        if path.name == "MANIFEST.md":
            continue
        lines.append("| `%s` | Clean SGCP experiment markdown package file. |" % path.name)
    lines.append("")
    (EXPERIMENT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = build_metric_rows()
    metric_csv = ARTIFACT / "table3_cov_scheduler_metrics_20260724.csv"
    write_csv(metric_csv, rows)
    compute_csv = run_compute_profile(metric_csv)
    compute_rows = load_compute(compute_csv)
    markdown = build_markdown(rows, compute_rows)
    (ARTIFACT / "table3_cov_scheduler_20260724.md").write_text(
        markdown, encoding="utf-8")
    (EXPERIMENT / "03_scheduler_comparison.md").write_text(
        markdown, encoding="utf-8")
    update_external_docs()
    write_manifest()
    print(metric_csv)
    print(compute_csv)
    print(EXPERIMENT / "03_scheduler_comparison.md")


if __name__ == "__main__":
    main()
