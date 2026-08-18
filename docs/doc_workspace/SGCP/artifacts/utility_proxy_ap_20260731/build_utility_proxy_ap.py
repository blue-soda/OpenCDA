# -*- coding: utf-8 -*-
"""Compute SGCP utility-surrogate/AP evidence without detector reruns.

The script replays the dense Table-3 schedulers, reads their actual grid
selection from the in-memory OpenCDA scheduler, computes the paper-facing
early/late utility proxy, and joins it with the already measured AP@0.5.
The utility uses the current paper formulation:

    U_early_r(g; x) = 1 - (1 - q_r(g)) prod_i (1 - x_i,r,g q_i(g)).

This is an offline analysis over existing traces/scheduler replay; it does not
run detector inference.
"""

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    apply_cluster_state,
    clear_sgcp_globals,
)
from opencda.core.clustering.algorithms.clustering.\
    potential_verified_cov_coalition_game import (
        PotentialVerifiedCOVCoalitionGame,
    )
from opencda.core.clustering.algorithms.resource_allocation import (
    build_resource_allocator,
)
from opencda.core.clustering.algorithms.resource_allocation.edgecooper import (
    edgecooper_grid_score,
)
from opencda.core.clustering.algorithms.resource_allocation.edgecooper_pmax \
    import trim_pmax_selection_to_global_deadline
from opencda.core.clustering.algorithms.resource_allocation.\
    selective_baseline_common import (
        EDGECOOPER_GLOBAL_COMM_RANGE_M,
        apply_receiver_grid_selection,
        collect_receiver_grid_selection,
    )
from opencda.core.clustering.algorithms.resource_allocation.\
    selective_baselines import (
        assign_selective_grid_selection,
        trim_selective_grid_selection_to_global_deadline,
    )
from opencda.core.clustering.utils.channel_model import build_channel_model
from opencda.tools.offline_inference import (
    apply_grid_selection_to_world,
    apply_resource_overrides,
    build_fixed_clusters,
    cluster_templates_from_clusters,
    fixed_delta_from_protocol,
    frame_interval_seconds,
    load_protocol,
    run_pcs_rounds_with_deadline,
    select_cav_ids,
    trim_grid_selection_to_deadline,
    trim_grid_selection_to_payload_budget,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = r"D:\Data\Carla"
SCENARIO_ID = "2026_07_29_02_32_08"
EGO_CAV_ID = 1
MAX_FRAMES = 41
UTILITY_SAMPLE_STEP = 4
RHO_TH = 2.0
BANDWIDTH_MHZ = 40.0
NUM_CHANNELS = 10
DATA_DEADLINE_MS = 60.0
HEAD_RB_BUDGET = 2
MAX_SENDERS_PER_RECEIVER = 2
SGCP_RAW_MBPS_BUDGET = 40.0


TABLE3_AP = {
    "SGCP": (0.86, 0.81, 0.58),
    "Cluster-head late only": (0.79, 0.71, 0.49),
    "FullPerception-PCS": (0.82, 0.76, 0.54),
    "Random budget": (0.82, 0.77, 0.53),
    "Density greedy": (0.81, 0.76, 0.52),
    "Link-aware density": (0.81, 0.76, 0.52),
    "PACP-LiDAR": (0.78, 0.70, 0.49),
    "EdgeCooper-HD": (0.81, 0.74, 0.51),
    "EdgeCooper-HD-Pmax": (0.88, 0.81, 0.56),
}


METHODS = [
    {"name": "SGCP", "kind": "sgcp", "ra": "dynamic_cv",
     "raw_budget": SGCP_RAW_MBPS_BUDGET, "cap": RHO_TH},
    {"name": "Cluster-head late only", "kind": "head_only"},
    {"name": "FullPerception-PCS", "kind": "pcs"},
    {"name": "Random budget", "kind": "selective", "baseline": "random"},
    {"name": "Density greedy", "kind": "selective", "baseline": "density"},
    {"name": "Link-aware density", "kind": "selective",
     "baseline": "communication_aware"},
    {"name": "PACP-LiDAR", "kind": "selective", "baseline": "pacp_lidar"},
    {"name": "EdgeCooper-HD", "kind": "selective",
     "baseline": "edgecooper_global_hd"},
    {"name": "EdgeCooper-HD-Pmax", "kind": "selective_pmax",
     "baseline": "edgecooper_global_hd_pmax", "cap": RHO_TH},
]


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), q))


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
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


def grid_quality(vm, grid_id, rho_th=RHO_TH):
    rho = max(float(rho_th), 1e-6)
    density = vm.perception_manager.lidar.get_grid_density(grid_id)
    return min(1.0, max(0.0, float(density) / rho))


def selected_raw_bytes(world, selection, cap_rho=None):
    total = 0
    for sender_grids in (selection or {}).values():
        for sender_id, grid_ids in sender_grids.items():
            sender_vm = world.get_vehicle_manager(int(sender_id))
            if sender_vm is None:
                continue
            for grid_id in grid_ids:
                points = sender_vm.perception_manager.lidar.\
                    get_local_points_by_grid_ids([grid_id])
                if points is None:
                    continue
                if cap_rho is not None and len(points) > 0:
                    # Approximate the density-capped deterministic upload:
                    # cap at the grid-level rho threshold while preserving
                    # point size. This is only for utility table diagnostics;
                    # payload in the paper table is from the trace itself.
                    point_size = int(points.nbytes / max(1, len(points)))
                    capped_points = min(len(points), int(math.ceil(cap_rho)))
                    total += capped_points * point_size
                else:
                    total += int(points.nbytes)
    return total


def utility_for_selection(world, clusters, selection, rho_th=RHO_TH):
    receiver_ids = sorted(int(cluster.head_id) for cluster in clusters)
    scene_grids = set()
    receiver_req = {}
    for receiver_id in receiver_ids:
        vm = world.get_vehicle_manager(receiver_id)
        req = set(vm.perception_manager.lidar.req_grids)
        receiver_req[receiver_id] = req
        scene_grids |= req
    if not scene_grids:
        return {
            "u_before": 0.0,
            "u_after": 0.0,
            "u_gain": 0.0,
            "cov_gain": 0.0,
            "view_gain": 0.0,
        }

    before_by_receiver = {}
    after_by_receiver = {}
    marginal_gain = 0.0
    for receiver_id in receiver_ids:
        receiver_vm = world.get_vehicle_manager(receiver_id)
        current = {}
        before = {}
        for grid_id in receiver_req[receiver_id]:
            q = grid_quality(receiver_vm, grid_id, rho_th=rho_th)
            current[grid_id] = q
            before[grid_id] = q
        for sender_id, grid_ids in sorted(
                (selection or {}).get(receiver_id, {}).items()):
            sender_vm = world.get_vehicle_manager(int(sender_id))
            if sender_vm is None:
                continue
            for grid_id in grid_ids:
                if grid_id not in receiver_req[receiver_id]:
                    continue
                qi = grid_quality(sender_vm, grid_id, rho_th=rho_th)
                if qi <= 0.0:
                    continue
                cur = current.get(grid_id, 0.0)
                delta = qi * max(0.0, 1.0 - cur)
                marginal_gain += delta
                current[grid_id] = min(1.0, cur + delta)
        before_by_receiver[receiver_id] = before
        after_by_receiver[receiver_id] = current

    u_before = 0.0
    u_after = 0.0
    for grid_id in scene_grids:
        u_before += max(
            before_by_receiver.get(receiver_id, {}).get(grid_id, 0.0)
            for receiver_id in receiver_ids)
        u_after += max(
            after_by_receiver.get(receiver_id, {}).get(grid_id, 0.0)
            for receiver_id in receiver_ids)
    denom = float(max(1, len(scene_grids)))
    return {
        "u_before": u_before / denom,
        "u_after": u_after / denom,
        "u_gain": (u_after - u_before) / denom,
        "marginal_gain": marginal_gain / denom,
    }


def load_world(dataset, protocol, scenario_id, timestamp):
    clear_sgcp_globals()
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=EGO_CAV_ID,
        cav_ids=select_cav_ids(dataset, scenario_id, ego_cav_id=EGO_CAV_ID))
    world = OfflineCavWorld(
        frame,
        ego_id=EGO_CAV_ID,
        protocol=protocol,
        density_threshold=RHO_TH)
    return world


def build_world_and_clusters(dataset, protocol, scenario_id, timestamp):
    world = load_world(dataset, protocol, scenario_id, timestamp)
    clustering = PotentialVerifiedCOVCoalitionGame(world)
    clustering.p.T_min_stab = frame_interval_seconds(
        [timestamp, "%06d" % (int(timestamp) + 20)],
        fixed_delta_from_protocol(protocol))
    clusters = clustering.run()
    apply_cluster_state(world, clusters)
    return world, clusters


def build_world_with_templates(dataset, protocol, scenario_id, timestamp,
                               templates):
    world = load_world(dataset, protocol, scenario_id, timestamp)
    clusters = build_fixed_clusters(world, templates)
    apply_cluster_state(world, clusters)
    return world, clusters


def run_sgcp_selection(world, clusters, channel_model, method):
    allocator = build_resource_allocator(method["ra"], world)
    apply_resource_overrides(
        allocator,
        world,
        num_channels=NUM_CHANNELS,
        bandwidth_mhz=BANDWIDTH_MHZ,
        head_rb_budget=HEAD_RB_BUDGET,
        channel_model=channel_model,
        max_senders_per_receiver=MAX_SENDERS_PER_RECEIVER)
    allocator.set_clusters(clusters)
    allocator.run()
    selection = collect_receiver_grid_selection(world, clusters)
    channel_allocation = {}
    for vm in world.get_vehicle_managers().values():
        scheduler = getattr(vm.v2x_manager, "scheduler", None)
        channel_allocation.update(
            getattr(scheduler, "channel_allocation", {}) or {})
    link_sc_nums = {
        (int(sender_id), int(receiver_id)): 1
        for (sender_id, receiver_id) in channel_allocation
    }
    selection = trim_grid_selection_to_deadline(
        world,
        selection,
        link_sc_nums,
        bandwidth_mhz=BANDWIDTH_MHZ,
        num_channels=NUM_CHANNELS,
        deadline_ms=DATA_DEADLINE_MS,
        channel_model=channel_model,
        upload_density_cap_rho=method.get("cap"))
    budget_bytes = int(float(method["raw_budget"]) * 1e6 * 0.1 / 8.0)
    selection, _, _ = trim_grid_selection_to_payload_budget(
        world,
        selection,
        budget_bytes,
        strategies=getattr(allocator, "strategies", None),
        upload_density_cap_rho=method.get("cap"))
    apply_grid_selection_to_world(
        world,
        selection,
        channel_allocation=channel_allocation)
    return collect_receiver_grid_selection(world, clusters)


def run_pcs_selection(world, clusters, channel_model):
    allocator = build_resource_allocator("fullperception_pcs", world)
    apply_resource_overrides(
        allocator,
        world,
        num_channels=NUM_CHANNELS,
        bandwidth_mhz=BANDWIDTH_MHZ,
        head_rb_budget=HEAD_RB_BUDGET,
        pcs_blind_spot_min_division=4,
        pcs_blind_spot_radius=4,
        pcs_min_spot_grids=128,
        pcs_communication_range_m=35.0,
        channel_model=channel_model,
        max_senders_per_receiver=MAX_SENDERS_PER_RECEIVER)
    allocator.set_clusters(clusters)
    run_pcs_rounds_with_deadline(
        allocator,
        world,
        max_rounds=1,
        deadline_ms=DATA_DEADLINE_MS,
        channel_model=channel_model)
    return collect_receiver_grid_selection(world, clusters)


def run_selective_selection(world, clusters, channel_model, method):
    baseline = method["baseline"]
    base_name = baseline.replace("_pmax", "")
    if base_name in ["edgecooper_global", "edgecooper_global_hd"]:
        world._edgecooper_global_sender_loads = {}
        world._edgecooper_global_cluster_count = len(clusters)
        world._edgecooper_global_receiver_ids = set(
            int(cluster.head_id) for cluster in clusters)
        world._edgecooper_global_comm_range_m = EDGECOOPER_GLOBAL_COMM_RANGE_M
    for cluster in clusters:
        assign_selective_grid_selection(
            world,
            cluster,
            base_name,
            member_budget=3,
            grid_budget=117,
            timestamp=None)
    if method["kind"] == "selective_pmax":
        admitted, _ = trim_pmax_selection_to_global_deadline(
            world,
            collect_receiver_grid_selection(world, clusters),
            baseline,
            DATA_DEADLINE_MS,
            channel_model,
            edgecooper_grid_score,
            max_senders_per_receiver=MAX_SENDERS_PER_RECEIVER,
            density_cap_rho=method.get("cap"))
        apply_receiver_grid_selection(world, clusters, admitted)
    else:
        trim_selective_grid_selection_to_global_deadline(
            world,
            clusters,
            base_name,
            DATA_DEADLINE_MS,
            channel_model,
            max_senders_per_receiver=MAX_SENDERS_PER_RECEIVER)
    return collect_receiver_grid_selection(world, clusters)


def selection_for_method(world, clusters, channel_model, method):
    if method["kind"] == "head_only":
        return {}
    if method["kind"] == "sgcp":
        return run_sgcp_selection(world, clusters, channel_model, method)
    if method["kind"] == "pcs":
        return run_pcs_selection(world, clusters, channel_model)
    if method["kind"] in ["selective", "selective_pmax"]:
        return run_selective_selection(world, clusters, channel_model, method)
    raise ValueError(method["kind"])


def summarize_rows(rows):
    out = {}
    for key in rows[0].keys():
        if key in ["method"]:
            continue
        values = [float(row[key]) for row in rows]
        out[key + "_mean"] = float(np.mean(values))
        out[key + "_p95"] = percentile(values, 95)
    return out


def main():
    dataset = OPV2VFrameDataset(DATASET_ROOT)
    protocol = load_protocol(dataset, SCENARIO_ID)
    timestamps = dataset.scenarios[SCENARIO_ID]["timestamps"][:MAX_FRAMES][
        ::UTILITY_SAMPLE_STEP]
    channel_model = build_channel_model(
        mode="ns3",
        bandwidth_mhz=BANDWIDTH_MHZ,
        num_channels=NUM_CHANNELS,
        frame_deadline_s=DATA_DEADLINE_MS / 1000.0,
        ns3_tb_size_bytes=899,
        ns3_slot_duration_ms=0.5,
        ns3_subchannel_prbs=10,
        ns3_symbols_per_slot=12,
        ns3_mcs=28)
    method_rows = []
    sample_rows = []
    template_by_timestamp = {}
    for timestamp in timestamps:
        _, clusters = build_world_and_clusters(
            dataset,
            protocol,
            SCENARIO_ID,
            timestamp)
        template_by_timestamp[timestamp] = cluster_templates_from_clusters(
            clusters)
    for method in METHODS:
        per_frame = []
        for timestamp in timestamps:
            world, clusters = build_world_with_templates(
                dataset,
                protocol,
                SCENARIO_ID,
                timestamp,
                template_by_timestamp[timestamp])
            selection = selection_for_method(world, clusters, channel_model,
                                             method)
            utility = utility_for_selection(world, clusters, selection)
            raw_bytes = selected_raw_bytes(
                world,
                selection,
                cap_rho=method.get("cap"))
            selected_grids = sum(
                len(grid_ids)
                for sender_grids in selection.values()
                for grid_ids in sender_grids.values())
            links = sum(
                1 for sender_grids in selection.values()
                for grid_ids in sender_grids.values() if grid_ids)
            row = {
                "method": method["name"],
                "timestamp": timestamp,
                "u_final": utility["u_after"],
                "u_gain": utility["u_gain"],
                "marginal_gain": utility["marginal_gain"],
                "selected_grids": float(selected_grids),
                "links": float(links),
                "raw_mbps": raw_bytes * 8.0 / 1e6 / 0.1,
            }
            sample_rows.append(row)
            per_frame.append(row)
        stats = summarize_rows(per_frame)
        ap03, ap05, ap07 = TABLE3_AP[method["name"]]
        method_rows.append({
            "method": method["name"],
            "utility_final": stats["u_final_mean"],
            "utility_gain": stats["u_gain_mean"],
            "marginal_gain": stats["marginal_gain_mean"],
            "selected_grids": stats["selected_grids_mean"],
            "links": stats["links_mean"],
            "raw_mbps_proxy": stats["raw_mbps_mean"],
            "ap_03": ap03,
            "ap_05": ap05,
            "ap_07": ap07,
        })

    csv_path = os.path.join(OUTPUT_DIR, "utility_proxy_vs_ap.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(method_rows[0].keys()))
        writer.writeheader()
        for row in method_rows:
            writer.writerow(row)

    sample_csv = os.path.join(OUTPUT_DIR, "utility_proxy_frame_samples.csv")
    with open(sample_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    ap = [row["ap_05"] for row in method_rows]
    gain = [row["utility_gain"] for row in method_rows]
    final = [row["utility_final"] for row in method_rows]
    selected = [row["selected_grids"] for row in method_rows]
    summary = {
        "pearson_utility_gain_ap05": pearson(gain, ap),
        "spearman_utility_gain_ap05": spearman(gain, ap),
        "pearson_utility_final_ap05": pearson(final, ap),
        "spearman_utility_final_ap05": spearman(final, ap),
        "pearson_selected_grids_ap05": pearson(selected, ap),
        "spearman_selected_grids_ap05": spearman(selected, ap),
    }
    with open(os.path.join(OUTPUT_DIR, "utility_proxy_summary.json"),
              "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    md_path = os.path.join(OUTPUT_DIR, "utility_proxy_vs_ap.md")
    with open(md_path, "w") as stream:
        stream.write("# Utility Proxy vs AP@0.5\n\n")
        stream.write(
            "Protocol: dense Table-3 SGCP-compatible scheduler comparison; "
            "11 utility-proxy validation frames sampled every 4 frames from "
            "the 41-frame evaluation window, 20 CAVs, PV coalition clustering, inter-cluster box "
            "NMS, 40 MHz / 10ch, 60 ms data-plane deadline. Utility is "
            "computed by replaying each scheduler without detector inference "
            "and evaluating the paper-facing early/late surrogate on the "
            "actual selected grid set.\n\n")
        stream.write("| Method | Utility final | Utility gain | Dynamic marginal gain | AP@0.5 | Avg grids | Links/frame |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(method_rows, key=lambda item: item["ap_05"],
                          reverse=True):
            stream.write(
                "| {method} | {utility_final:.4f} | {utility_gain:.4f} | "
                "{marginal_gain:.4f} | {ap_05:.2f} | "
                "{selected_grids:.2f} | {links:.2f} |\n".format(**row))
        stream.write("\nCorrelation with AP@0.5:\n\n")
        stream.write(
            "- Utility gain: Pearson `{:.3f}`, Spearman `{:.3f}`.\n".format(
                summary["pearson_utility_gain_ap05"],
                summary["spearman_utility_gain_ap05"]))
        stream.write(
            "- Final utility: Pearson `{:.3f}`, Spearman `{:.3f}`.\n".format(
                summary["pearson_utility_final_ap05"],
                summary["spearman_utility_final_ap05"]))
        stream.write(
            "- Selected grid count alone: Pearson `{:.3f}`, Spearman `{:.3f}`.\n".format(
                summary["pearson_selected_grids_ap05"],
                summary["spearman_selected_grids_ap05"]))
        stream.write(
            "\nInterpretation: if utility correlation exceeds selected-grid "
            "correlation, the density-derived C/V surrogate explains AP "
            "better than a pure communication-volume proxy.\n")

    print("wrote", csv_path)
    print("wrote", md_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
