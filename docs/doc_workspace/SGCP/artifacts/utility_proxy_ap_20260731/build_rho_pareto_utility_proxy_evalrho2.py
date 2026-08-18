# -*- coding: utf-8 -*-
"""Recompute rho-Pareto utility values on a fixed rho=2 scale.

The dense parameter table varies rho_th and raw-LiDAR budget.  The original
``build_rho_pareto_utility_proxy.py`` evaluates each row on its own rho scale,
which is useful for reproducing the scheduler but makes cross-rho utility
values harder to compare.  This script keeps each row's scheduler replay and
reported AP/Mbps intact, but evaluates the resulting selected grids with
``rho_eval = 2`` so every row has the same utility scale.
"""

import csv
import importlib.util
import json
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
UTILITY_DIR = (
    REPO / "docs/doc_workspace/SGCP/artifacts/utility_proxy_ap_20260731")
BASE_SCRIPT = UTILITY_DIR / "build_rho_pareto_utility_proxy.py"
SCHEDULER_CSV = UTILITY_DIR / "utility_proxy_vs_ap.csv"
OUT_CSV = UTILITY_DIR / "rho_pareto_utility_proxy_evalrho2_vs_ap.csv"
OUT_JSON = UTILITY_DIR / "rho_pareto_utility_proxy_evalrho2_summary.json"
OUT_COMBINED_CSV = (
    UTILITY_DIR / "combined_scheduler_rhopareto_evalrho2_vs_ap.csv")
OUT_COMBINED_JSON = (
    UTILITY_DIR / "combined_scheduler_rhopareto_evalrho2_summary.json")

RHO_EVAL = 2.0

TABLE3_AP = {
    "SGCP": 0.81,
    "Cluster-head late only": 0.71,
    "FullPerception-PCS": 0.76,
    "Random budget": 0.77,
    "Density greedy": 0.76,
    "Link-aware density": 0.76,
    "PACP-LiDAR": 0.70,
    "EdgeCooper-HD": 0.74,
    "EdgeCooper-HD-Pmax": 0.81,
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("rho_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(rows, key):
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def summary_for(rows):
    ap05 = [float(row["ap_05"]) for row in rows]
    summary = {
        "n_rows": len(rows),
        "rho_eval": RHO_EVAL,
        "pearson_utility_gain_ap05": base.pearson(
            [float(row["utility_gain"]) for row in rows], ap05),
        "spearman_utility_gain_ap05": base.spearman(
            [float(row["utility_gain"]) for row in rows], ap05),
        "pearson_final_utility_ap05": base.pearson(
            [float(row["utility_final"]) for row in rows], ap05),
        "spearman_final_utility_ap05": base.spearman(
            [float(row["utility_final"]) for row in rows], ap05),
        "pearson_selected_grids_ap05": base.pearson(
            [float(row["avg_selected_grids"]) for row in rows], ap05),
        "spearman_selected_grids_ap05": base.spearman(
            [float(row["avg_selected_grids"]) for row in rows], ap05),
    }
    rows_with_mbps = [
        row for row in rows
        if row.get("total_mbps", "") not in ["", None]
    ]
    if len(rows_with_mbps) == len(rows):
        summary["pearson_total_mbps_ap05"] = base.pearson(
            [float(row["total_mbps"]) for row in rows], ap05)
        summary["spearman_total_mbps_ap05"] = base.spearman(
            [float(row["total_mbps"]) for row in rows], ap05)
    return summary


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    global base
    base = load_base_module()
    utility = base.load_utility_module()

    parameter_rows = base.parse_parameter_table()
    budget_index = base.build_budget_index()
    matched = []
    for row in parameter_rows:
        item = base.match_budget(row, budget_index)
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
                templates[rho][timestamp] = (
                    utility.cluster_templates_from_clusters(clusters))

        method = {
            "name": "SGCP rho %.0f budget %.0f" % (
                rho, row["budget_mbps"]),
            "kind": "sgcp",
            "ra": "dynamic_cv",
            "raw_budget": row["budget_mbps"],
            "cap": rho,
        }
        per_frame = []
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
                world, clusters, selection, rho_th=RHO_EVAL))

        row["rho_eval"] = RHO_EVAL
        row["utility_final"] = summarize(per_frame, "u_after")
        row["utility_gain"] = summarize(per_frame, "u_gain")
        row["dynamic_marginal_gain"] = summarize(per_frame, "marginal_gain")
        rows_out.append(row)
        print(
            "rho=%s eval=%.0f budget=%s raw=%.2f utility_gain=%.4f ap05=%.2f"
            % (row["rho_th"], RHO_EVAL, row["budget_mbps"],
               row["raw_lidar_mbps"], row["utility_gain"], row["ap_05"]))

    fields = [
        "rho_th", "rho_eval", "budget_mbps", "raw_lidar_mbps",
        "total_mbps", "utility_final", "utility_gain",
        "dynamic_marginal_gain", "ap_03", "ap_05", "ap_07",
        "gflops_frame", "avg_source_cavs", "avg_selected_grids", "artifact",
    ]
    write_csv(OUT_CSV, rows_out, fields)
    OUT_JSON.write_text(
        json.dumps(summary_for(rows_out), indent=2, sort_keys=True),
        encoding="utf-8")

    combined = []
    with SCHEDULER_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            combined.append({
                "source": "scheduler_variant",
                "label": row["method"],
                "rho_eval": RHO_EVAL,
                "utility_final": float(row["utility_final"]),
                "utility_gain": float(row["utility_gain"]),
                "total_mbps": "",
                "avg_selected_grids": float(row["selected_grids"]),
                "ap_05": TABLE3_AP[row["method"]],
            })
    for row in rows_out:
        # Exclude the saturated rho=2 point because it duplicates the SGCP
        # scheduler row under a slightly different AP table rounding.
        if abs(float(row["rho_th"]) - 2.0) < 1e-9 and float(
                row["budget_mbps"]) == 40.0:
            continue
        combined.append({
            "source": "rho_pareto_evalrho2",
            "label": "rho=%.0f budget=%.0f Mbps" % (
                row["rho_th"], row["budget_mbps"]),
            "rho_eval": RHO_EVAL,
            "utility_final": row["utility_final"],
            "utility_gain": row["utility_gain"],
            "total_mbps": row["total_mbps"],
            "avg_selected_grids": row["avg_selected_grids"],
            "ap_05": row["ap_05"],
        })

    combined_fields = [
        "source", "label", "rho_eval", "utility_final", "utility_gain",
        "total_mbps", "avg_selected_grids", "ap_05",
    ]
    write_csv(OUT_COMBINED_CSV, combined, combined_fields)
    OUT_COMBINED_JSON.write_text(
        json.dumps(summary_for(combined), indent=2, sort_keys=True),
        encoding="utf-8")

    print("wrote", OUT_CSV)
    print("wrote", OUT_JSON)
    print("wrote", OUT_COMBINED_CSV)
    print("wrote", OUT_COMBINED_JSON)
    print("rho pareto", OUT_JSON.read_text(encoding="utf-8"))
    print("combined", OUT_COMBINED_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
