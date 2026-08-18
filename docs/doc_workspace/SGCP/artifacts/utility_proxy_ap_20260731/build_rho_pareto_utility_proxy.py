# -*- coding: utf-8 -*-
"""Compute utility/AP correlation for the dense rho-Pareto table.

This complements ``build_utility_proxy_ap.py``.  The existing utility proxy
table is scheduler-level; this script keeps the scheduler fixed to SGCP and
uses the rows reported in ``06_parameter_sensitivity.md``.
"""

import csv
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import numpy as np


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT_DIR = REPO / "docs/doc_workspace/SGCP/artifacts"
UTILITY_DIR = ARTIFACT_DIR / "utility_proxy_ap_20260731"
SENS_DIR = ARTIFACT_DIR / "dense_dynamic_cv_sensitivity_20260729"
CLEAN_MD = Path(
    r"C:\Workspace\2026-7-papers\infocom\SGCP"
    r"\experiment-dense-lidar-ver\06_parameter_sensitivity.md")
OUT_CSV = UTILITY_DIR / "rho_pareto_utility_proxy_vs_ap.csv"
OUT_JSON = UTILITY_DIR / "rho_pareto_utility_proxy_summary.json"
OUT_MD = UTILITY_DIR / "rho_pareto_utility_proxy_vs_ap.md"


def load_utility_module():
    script = UTILITY_DIR / "build_utility_proxy_ap.py"
    spec = importlib.util.spec_from_file_location("utility_proxy_ap", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_parameter_table():
    rows = []
    pattern = re.compile(
        r"^\|\s*(\d+(?:\.\d+)?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*"
        r"\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*"
        r"\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
    for line in CLEAN_MD.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        rho, raw, total, ap03, ap05, ap07, gflops, sources, grids = [
            float(item) for item in match.groups()
        ]
        rows.append({
            "rho_th": rho,
            "raw_lidar_mbps": raw,
            "total_mbps": total,
            "ap_03": ap03,
            "ap_05": ap05,
            "ap_07": ap07,
            "gflops_frame": gflops,
            "avg_source_cavs": sources,
            "avg_selected_grids": grids,
        })
    return rows


def parse_budget_from_name(name):
    match = re.match(r"budget_rho([0-9]+)_mbps([0-9]+)$", name)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def trace_raw_mbps(trace_path):
    with trace_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    frames = len({row["timestamp"] for row in rows})
    if frames <= 0:
        return 0.0
    raw_bytes = sum(int(float(row.get("communication_bytes") or 0))
                    for row in rows)
    return raw_bytes * 8.0 / (frames * 0.1) / 1e6


def parse_ap(out_path):
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"IOU 0\.3 is ([0-9.]+).*?IOU 0\.5 is ([0-9.]+).*?"
        r"IOU 0\.7 is ([0-9.]+)",
        text,
        re.S)
    if not matches:
        return None
    return tuple(float(item) for item in matches[-1])


def build_budget_index():
    index = []
    for directory in sorted(SENS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        parsed = parse_budget_from_name(directory.name)
        if parsed is None:
            continue
        trace = directory / "trace.csv"
        out = directory / "run.out"
        if not trace.exists() or not out.exists():
            continue
        ap = parse_ap(out)
        if ap is None:
            continue
        rho, budget = parsed
        index.append({
            "dir": directory,
            "rho_th": rho,
            "budget_mbps": budget,
            "raw_lidar_mbps": trace_raw_mbps(trace),
            "ap_03": ap[0],
            "ap_05": ap[1],
            "ap_07": ap[2],
        })
    return index


def match_budget(row, budget_index):
    candidates = [
        item for item in budget_index
        if abs(item["rho_th"] - row["rho_th"]) < 1e-6
    ]
    if not candidates:
        raise RuntimeError("No budget candidates for rho=%s" % row["rho_th"])
    candidates.sort(key=lambda item: (
        abs(item["raw_lidar_mbps"] - row["raw_lidar_mbps"]),
        abs(item["ap_05"] - row["ap_05"]),
        item["budget_mbps"]))
    best = candidates[0]
    if abs(best["raw_lidar_mbps"] - row["raw_lidar_mbps"]) > 1.0:
        raise RuntimeError(
            "No close match for rho=%s raw=%.2f; best %s raw=%.2f" % (
                row["rho_th"], row["raw_lidar_mbps"],
                best["dir"].name, best["raw_lidar_mbps"]))
    return best


def pearson(xs, ys):
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def summarize(rows, key):
    values = [float(row[key]) for row in rows]
    return float(np.mean(np.array(values, dtype=float)))


def main():
    utility = load_utility_module()
    parameter_rows = parse_parameter_table()
    budget_index = build_budget_index()
    matched = []
    for row in parameter_rows:
        item = match_budget(row, budget_index)
        merged = dict(row)
        merged["budget_mbps"] = item["budget_mbps"]
        merged["artifact"] = item["dir"].name
        matched.append(merged)

    dataset = utility.OPV2VFrameDataset(utility.DATASET_ROOT)
    protocol = utility.load_protocol(dataset, utility.SCENARIO_ID)
    timestamps = dataset.scenarios[utility.SCENARIO_ID]["timestamps"][
        :utility.MAX_FRAMES][::utility.UTILITY_SAMPLE_STEP]
    channel_model = utility.build_channel_model(
        mode="ns3",
        bandwidth_mhz=utility.BANDWIDTH_MHZ,
        num_channels=utility.NUM_CHANNELS,
        frame_deadline_s=utility.DATA_DEADLINE_MS / 1000.0,
        ns3_tb_size_bytes=899,
        ns3_slot_duration_ms=0.5,
        ns3_subchannel_prbs=10,
        ns3_symbols_per_slot=12,
        ns3_mcs=28)

    templates = {}
    rows_out = []
    for row in matched:
        rho = float(row["rho_th"])
        utility.RHO_TH = rho
        if rho not in templates:
            templates[rho] = {}
            for timestamp in timestamps:
                _, clusters = utility.build_world_and_clusters(
                    dataset, protocol, utility.SCENARIO_ID, timestamp)
                templates[rho][timestamp] = utility.cluster_templates_from_clusters(
                    clusters)
        per_frame = []
        method = {
            "name": "SGCP rho %.0f budget %.0f" % (
                rho, row["budget_mbps"]),
            "kind": "sgcp",
            "ra": "dynamic_cv",
            "raw_budget": row["budget_mbps"],
            "cap": rho,
        }
        for timestamp in timestamps:
            utility.RHO_TH = rho
            world, clusters = utility.build_world_with_templates(
                dataset,
                protocol,
                utility.SCENARIO_ID,
                timestamp,
                templates[rho][timestamp])
            selection = utility.selection_for_method(
                world, clusters, channel_model, method)
            per_frame.append(utility.utility_for_selection(
                world, clusters, selection, rho_th=rho))

        row["utility_final"] = summarize(per_frame, "u_after")
        row["utility_gain"] = summarize(per_frame, "u_gain")
        row["dynamic_marginal_gain"] = summarize(per_frame, "marginal_gain")
        rows_out.append(row)
        print("rho=%s budget=%s raw=%.2f utility_gain=%.4f ap05=%.2f" % (
            row["rho_th"], row["budget_mbps"], row["raw_lidar_mbps"],
            row["utility_gain"], row["ap_05"]))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "rho_th", "budget_mbps", "raw_lidar_mbps", "total_mbps",
            "utility_final", "utility_gain", "dynamic_marginal_gain",
            "ap_03", "ap_05", "ap_07", "gflops_frame",
            "avg_source_cavs", "avg_selected_grids", "artifact",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ap05 = [row["ap_05"] for row in rows_out]
    summary = {
        "n_rows": len(rows_out),
        "pearson_utility_gain_ap05": pearson(
            [row["utility_gain"] for row in rows_out], ap05),
        "spearman_utility_gain_ap05": spearman(
            [row["utility_gain"] for row in rows_out], ap05),
        "pearson_final_utility_ap05": pearson(
            [row["utility_final"] for row in rows_out], ap05),
        "spearman_final_utility_ap05": spearman(
            [row["utility_final"] for row in rows_out], ap05),
        "pearson_total_mbps_ap05": pearson(
            [row["total_mbps"] for row in rows_out], ap05),
        "spearman_total_mbps_ap05": spearman(
            [row["total_mbps"] for row in rows_out], ap05),
        "pearson_selected_grids_ap05": pearson(
            [row["avg_selected_grids"] for row in rows_out], ap05),
        "spearman_selected_grids_ap05": spearman(
            [row["avg_selected_grids"] for row in rows_out], ap05),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True),
                        encoding="utf-8")

    with OUT_MD.open("w", encoding="utf-8") as stream:
        stream.write("# Rho-Pareto Utility Proxy vs AP@0.5\n\n")
        stream.write(
            "This analysis uses the rows from the dense "
            "`06_parameter_sensitivity.md` rho Pareto table. The scheduler is "
            "fixed to SGCP; only `rho_th` and the raw-LiDAR budget vary. "
            "Utility is recomputed by replaying the SGCP scheduler on the "
            "same 11 validation frames used by the scheduler-level utility "
            "diagnostic.\n\n")
        stream.write("| rho_th | Budget Mbps | Raw Mbps | Total Mbps | Utility gain | Final utility | AP@0.5 |\n")
        stream.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows_out:
            stream.write(
                "| {rho_th:.0f} | {budget_mbps:.0f} | "
                "{raw_lidar_mbps:.2f} | {total_mbps:.2f} | "
                "{utility_gain:.4f} | {utility_final:.4f} | "
                "{ap_05:.2f} |\n".format(**row))
        stream.write("\nCorrelation with AP@0.5:\n\n")
        stream.write("| Explanatory variable | Pearson | Spearman |\n")
        stream.write("| --- | ---: | ---: |\n")
        stream.write("| Utility gain | %.3f | %.3f |\n" % (
            summary["pearson_utility_gain_ap05"],
            summary["spearman_utility_gain_ap05"]))
        stream.write("| Final utility | %.3f | %.3f |\n" % (
            summary["pearson_final_utility_ap05"],
            summary["spearman_final_utility_ap05"]))
        stream.write("| Total Mbps | %.3f | %.3f |\n" % (
            summary["pearson_total_mbps_ap05"],
            summary["spearman_total_mbps_ap05"]))
        stream.write("| Avg selected grids | %.3f | %.3f |\n" % (
            summary["pearson_selected_grids_ap05"],
            summary["spearman_selected_grids_ap05"]))

    print("wrote", OUT_CSV)
    print("wrote", OUT_MD)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
