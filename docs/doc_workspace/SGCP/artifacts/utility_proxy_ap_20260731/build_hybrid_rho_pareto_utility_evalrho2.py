# -*- coding: utf-8 -*-
"""Utility/AP correlation for the hybrid rho-Pareto addendum.

This script replays the current hybrid SGCP scheduler
``hybrid_round_robin_dynamic_marginal`` for every row in the dense hybrid
rho-Pareto table.  Each row keeps its experiment rho/budget for scheduling and
payload capping, but the utility values are evaluated on the formal
``rho_eval = 2`` scale so all rows are comparable.
"""

import csv
import importlib.util
import json
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
UTILITY_DIR = (
    REPO / "docs/doc_workspace/SGCP/artifacts/utility_proxy_ap_20260731")
BASE_SCRIPT = UTILITY_DIR / "build_rho_pareto_utility_proxy.py"
HYBRID_CSV = (
    REPO / "docs/doc_workspace/SGCP/artifacts/hybrid_followup_20260801"
    / "rho_pareto/hybrid_rho_pareto.csv")
SCHEDULER_CSV = UTILITY_DIR / "utility_proxy_vs_ap.csv"

OUT_CSV = UTILITY_DIR / "hybrid_rho_pareto_utility_evalrho2_vs_ap.csv"
OUT_JSON = UTILITY_DIR / "hybrid_rho_pareto_utility_evalrho2_summary.json"
OUT_COMBINED_CSV = (
    UTILITY_DIR / "combined_scheduler_hybrid_evalrho2_vs_ap.csv")
OUT_COMBINED_JSON = (
    UTILITY_DIR / "combined_scheduler_hybrid_evalrho2_summary.json")

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


def summary_for(rows, include_total_mbps=True):
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
    if include_total_mbps:
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


def load_hybrid_rows():
    with HYBRID_CSV.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            rows.append({
                "rho_th": float(row["rho_th"]),
                "raw_budget_mbps": float(row["raw_budget_mbps"]),
                "raw_mbps": float(row["raw_mbps"]),
                "box_mbps": float(row["box_mbps"]),
                "total_mbps": float(row["total_mbps"]),
                "ap_03": float(row["ap_03"]),
                "ap_05": float(row["ap_05"]),
                "ap_07": float(row["ap_07"]),
                "avg_source_cavs": float(row["avg_source_cavs"]),
                "avg_selected_grids": float(row["avg_selected_grids"]),
                "p95_data_time_ms": float(row["p95_data_time_ms"]),
                "trace": row["trace"],
            })
    rows.sort(key=lambda item: (item["rho_th"], item["raw_budget_mbps"]))
    return rows


def main():
    global base
    base = load_base_module()
    utility = base.load_utility_module()

    hybrid_rows = load_hybrid_rows()
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
    for row in hybrid_rows:
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
            "name": "Hybrid SGCP rho %.0f budget %.0f" % (
                rho, row["raw_budget_mbps"]),
            "kind": "sgcp",
            "ra": "hybrid_round_robin_dynamic_marginal",
            "raw_budget": row["raw_budget_mbps"],
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

        out = dict(row)
        out["rho_eval"] = RHO_EVAL
        out["utility_final"] = summarize(per_frame, "u_after")
        out["utility_gain"] = summarize(per_frame, "u_gain")
        out["dynamic_marginal_gain"] = summarize(per_frame, "marginal_gain")
        rows_out.append(out)
        print(
            "hybrid rho=%s eval=%.0f budget=%s raw=%.2f utility_gain=%.4f ap05=%.2f"
            % (out["rho_th"], RHO_EVAL, out["raw_budget_mbps"],
               out["raw_mbps"], out["utility_gain"], out["ap_05"]))

    fields = [
        "rho_th", "rho_eval", "raw_budget_mbps", "raw_mbps", "box_mbps",
        "total_mbps", "utility_final", "utility_gain",
        "dynamic_marginal_gain", "ap_03", "ap_05", "ap_07",
        "avg_source_cavs", "avg_selected_grids", "p95_data_time_ms", "trace",
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
        # Exclude the saturated formal rho=2 row because it duplicates the
        # SGCP operating point represented by the scheduler-variant row, under
        # the updated hybrid implementation and rounded AP table.
        if abs(float(row["rho_th"]) - 2.0) < 1e-9 and float(
                row["raw_budget_mbps"]) == 40.0:
            continue
        combined.append({
            "source": "hybrid_rho_pareto_evalrho2",
            "label": "rho=%.0f budget=%.0f Mbps" % (
                row["rho_th"], row["raw_budget_mbps"]),
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
        json.dumps(
            summary_for(combined, include_total_mbps=False),
            indent=2,
            sort_keys=True),
        encoding="utf-8")

    print("wrote", OUT_CSV)
    print("wrote", OUT_JSON)
    print("wrote", OUT_COMBINED_CSV)
    print("wrote", OUT_COMBINED_JSON)
    print("hybrid", OUT_JSON.read_text(encoding="utf-8"))
    print("combined", OUT_COMBINED_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
