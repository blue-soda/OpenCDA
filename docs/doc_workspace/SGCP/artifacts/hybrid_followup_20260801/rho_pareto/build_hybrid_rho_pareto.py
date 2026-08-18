# -*- coding: utf-8 -*-
"""Build the hybrid SGCP rho Pareto table from completed runs."""

import csv
import math
import re
from pathlib import Path


ARTIFACT = Path(
    r"C:\Workspace\OpenCDA\docs\doc_workspace\SGCP\artifacts"
    r"\hybrid_followup_20260801\rho_pareto")
EXPERIMENT = Path(
    r"C:\Workspace\2026-7-papers\infocom\SGCP"
    r"\experiment-dense-lidar-ver")

AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is\s+([0-9.]+).*?"
    r"IOU 0\.5 is\s+([0-9.]+).*?"
    r"IOU 0\.7 is\s+([0-9.]+)")
SUMMARY_RE = re.compile(
    r"sgcp_summary\s+frames=(\d+)\s+avg_comm_bytes=([0-9.]+)\s+"
    r"total_comm_bytes=(\d+)\s+avg_source_cavs=([0-9.]+)\s+"
    r"avg_selected_grids=([0-9.]+)")
BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1


def parse_tag(path):
    match = re.match(r"rho(?P<rho>[0-9p]+)_budget(?P<budget>[0-9p]+)\.out$",
                     path.name)
    if not match:
        return None
    rho = float(match.group("rho").replace("p", "."))
    budget = float(match.group("budget").replace("p", "."))
    return rho, budget


def parse_run(log_path):
    tag = parse_tag(log_path)
    if tag is None:
        return None
    trace_path = log_path.with_name(log_path.stem + "_trace.csv")
    eval_path = log_path.with_name(log_path.stem + "_eval_stats.csv")
    if not trace_path.exists() or not eval_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\x00", "")
    aps = AP_RE.findall(text)
    summaries = SUMMARY_RE.findall(text)
    if not aps or not summaries:
        return None
    ap_03, ap_05, ap_07 = [float(v) for v in aps[-1]]
    frames_s, _, total_bytes_s, avg_sources_s, avg_grids_s = summaries[-1]
    frames_from_log = int(frames_s)
    total_bytes = int(total_bytes_s)

    rows = list(csv.DictReader(trace_path.open(newline="", encoding="utf-8")))
    timestamps = sorted({row.get("timestamp", "") for row in rows
                         if row.get("timestamp", "")})
    frames = len(timestamps) or max(frames_from_log, 1)
    by_sample = {}
    frame_times = []
    for ts in timestamps:
        frame_times.append(max(
            float(row.get("frame_comm_time_ms") or 0.0)
            for row in rows if row.get("timestamp") == ts))
    for row in rows:
        key = (row.get("timestamp", ""), row.get("receiver_id", ""))
        if key[0] and key[1]:
            boxes = int(float(row.get("pred_boxes") or 0))
            by_sample[key] = max(by_sample.get(key, 0), boxes)
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values()
        if boxes > 0)
    duration_s = frames * FRAME_INTERVAL_S
    frame_times_sorted = sorted(frame_times)
    p95 = 0.0
    if frame_times_sorted:
        p95 = frame_times_sorted[math.ceil(0.95 * len(frame_times_sorted)) - 1]
    rho, budget = tag
    return {
        "rho_th": rho,
        "raw_budget_mbps": budget,
        "raw_mbps": total_bytes * 8.0 / duration_s / 1e6,
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
        "total_mbps": (total_bytes + box_bytes) * 8.0 / duration_s / 1e6,
        "ap_03": ap_03,
        "ap_05": ap_05,
        "ap_07": ap_07,
        "avg_source_cavs": float(avg_sources_s),
        "avg_selected_grids": float(avg_grids_s),
        "p95_data_time_ms": p95,
        "max_data_time_ms": max(frame_times) if frame_times else 0.0,
        "trace": str(trace_path),
    }


def fmt(value):
    return "%.2f" % float(value)


def main():
    rows = []
    for log_path in sorted(ARTIFACT.glob("rho*_budget*.out")):
        row = parse_run(log_path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: (row["rho_th"], row["raw_budget_mbps"]))
    if not rows:
        raise SystemExit("No completed rho runs found.")

    csv_path = ARTIFACT / "hybrid_rho_pareto.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "## Hybrid Scheduler Rho Pareto Addendum",
        "",
        "Status: generated from completed hybrid rho sweep runs only. If the "
        "background sweep is still running, this file is an intermediate "
        "snapshot and must be regenerated before paper use.",
        "",
        "Protocol: same dense Town03 full-frame-GT protocol as Table 1, but "
        "with `hybrid_round_robin_dynamic_marginal`; `rho_th` and "
        "`upload_density_cap_rho` are varied together. These rows are appended "
        "as SGCP hybrid evidence and do not replace the previous `dynamic_cv` "
        "Pareto table.",
        "",
        "| rho_th | Raw budget Mbps | Raw Mbps | Box Mbps | Total Mbps | AP@0.3 | AP@0.5 | AP@0.7 | Avg source CAVs | Avg selected grids | P95 data time |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {rho} | {budget} | {raw} | {box} | {total} | {a3} | "
            "{a5} | {a7} | {src} | {grids} | {p95} ms |".format(
                rho=fmt(row["rho_th"]),
                budget=fmt(row["raw_budget_mbps"]),
                raw=fmt(row["raw_mbps"]),
                box=fmt(row["box_mbps"]),
                total=fmt(row["total_mbps"]),
                a3=fmt(row["ap_03"]),
                a5=fmt(row["ap_05"]),
                a7=fmt(row["ap_07"]),
                src=fmt(row["avg_source_cavs"]),
                grids=fmt(row["avg_selected_grids"]),
                p95=fmt(row["p95_data_time_ms"]),
            )
        )
    best = max(rows, key=lambda row: (row["ap_05"], row["ap_03"],
                                      -row["total_mbps"]))
    lines.extend([
        "",
        "Current best completed hybrid row by AP@0.5/AP@0.3 is "
        "`rho_th=%.2f`, raw budget `%.2f Mbps`, AP `%.2f/%.2f/%.2f`, "
        "total payload `%.2f Mbps`." % (
            best["rho_th"], best["raw_budget_mbps"], best["ap_03"],
            best["ap_05"], best["ap_07"], best["total_mbps"]),
        "",
        "Provenance CSV: `%s`." % csv_path,
    ])
    md_path = ARTIFACT / "hybrid_rho_pareto.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target = EXPERIMENT / "06_parameter_sensitivity_hybrid_addendum.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)
    print(target)
    print("completed_rows=%d" % len(rows))


if __name__ == "__main__":
    main()
