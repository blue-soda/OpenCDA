# -*- coding: utf-8 -*-
"""Profile cold-start vs warm-start PV coalition convergence.

This script is intentionally artifact-local: it does not change SGCP runtime
code.  It replays the 41-frame offline dump and compares per-frame coalition
formation from singleton initialization with maintenance from the previous
frame's final partition.
"""

from __future__ import print_function

import csv
import json
import os
import statistics
import time

import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    clear_sgcp_globals,
)
from opencda.core.clustering.algorithms.clustering.\
    potential_verified_cov_coalition_game import (
        PotentialVerifiedCOVCoalitionGame,
    )
from opencda.core.clustering.utils import common


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../.."))
OUTPUT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DATASET_ROOT = r"D:\Data\Carla"
SCENARIO_ID = "2026_07_15_01_26_56"
EGO_CAV_ID = 1
FRAME_START = 60
FRAME_END = 140
FRAME_STEP = 2
N_MAX = 4
RHO_TH = 3.0
T_MIN_STAB = 0.1
MAX_ITER = 20


class CountedPotentialVerifiedCOVCoalitionGame(
        PotentialVerifiedCOVCoalitionGame):
    """PV coalition game with convergence counters."""

    def __init__(self, cav_world):
        super(CountedPotentialVerifiedCOVCoalitionGame, self).__init__(
            cav_world)
        self.profile = {
            "rounds": 0,
            "accepted_migrations": 0,
            "potential_checks": 0,
            "proxy_accepts": 0,
            "elapsed_ms": 0.0,
        }

    def coalition_formation(self, max_iter=20):
        start = time.perf_counter()
        self.check_is_ok()
        self.ego_coalition_be_first()
        self.capacity_stats = {
            "full_candidate_skips": 0,
        }
        self.last_potential_checks = []
        rounds = 0
        accepted_migrations = 0
        potential_checks = 0
        proxy_accepts = 0
        for iteration in range(max_iter):
            rounds = iteration + 1
            updated = False
            for vid in list(common.global_vehicles.keys()):
                current = self.find_coalition(vid)
                if current is None:
                    continue
                current_contribution = self.current_contribution(current, vid)
                best_candidate = None
                for coalition in list(self.coalitions):
                    if coalition is current:
                        continue
                    if coalition.size() >= self.p.N_max:
                        self.capacity_stats["full_candidate_skips"] += 1
                        continue
                    delta = self.marginal_contribution(coalition, vid)
                    proxy_accept = delta > current_contribution * self.p.ita
                    phi_delta, phi_before, phi_after = (
                        self.affected_potential_delta(vid, current, coalition))
                    potential_checks += 1
                    if proxy_accept:
                        proxy_accepts += 1
                    check = {
                        "vehicle_id": int(vid),
                        "source_members": sorted(
                            int(item) for item in current.members),
                        "target_members": sorted(
                            int(item) for item in coalition.members),
                        "proxy_before": float(current_contribution),
                        "proxy_after": float(delta),
                        "phi_before": float(phi_before),
                        "phi_after": float(phi_after),
                        "phi_delta": float(phi_delta),
                        "proxy_accept": bool(proxy_accept),
                        "accepted": False,
                    }
                    self.last_potential_checks.append(check)
                    if not proxy_accept or phi_delta <= 1e-9:
                        continue
                    candidate = {
                        "coalition": coalition,
                        "proxy_after": delta,
                        "phi_delta": phi_delta,
                        "check": check,
                    }
                    if best_candidate is None:
                        best_candidate = candidate
                    elif (delta, phi_delta) > (
                            best_candidate["proxy_after"],
                            best_candidate["phi_delta"]):
                        best_candidate = candidate

                if best_candidate is None:
                    continue

                best_coalition = best_candidate["coalition"]
                best_candidate["check"]["accepted"] = True
                current.remove_member(vid)
                best_coalition.add_member(vid)
                accepted_migrations += 1
                if current.size() == 0:
                    self.coalitions.remove(current)
                    current = None
                if current is not None and current.head_id in current.members:
                    current.grid_bits = current.compute_grid_bits()
                if best_coalition.head_id in best_coalition.members:
                    best_coalition.grid_bits = best_coalition.compute_grid_bits()
                updated = True
            if not updated:
                break

        self.profile = {
            "rounds": rounds,
            "accepted_migrations": accepted_migrations,
            "potential_checks": potential_checks,
            "proxy_accepts": proxy_accepts,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        }
        return self.coalitions


def load_protocol(dataset, scenario_id):
    protocol_path = os.path.join(
        dataset.scenarios[scenario_id]["path"],
        "data_protocol.yaml")
    with open(protocol_path, "r") as stream:
        return yaml.load(stream, Loader=yaml.Loader) or {}


def selected_timestamps(dataset, scenario_id):
    timestamps = dataset.scenarios[scenario_id]["timestamps"]
    selected = []
    for timestamp in timestamps:
        try:
            frame_id = int(timestamp)
        except ValueError:
            continue
        if FRAME_START <= frame_id <= FRAME_END and (
                frame_id - FRAME_START) % FRAME_STEP == 0:
            selected.append(timestamp)
    return selected


def canonical_partition(clusters):
    return tuple(sorted(
        tuple(sorted(int(member) for member in cluster.members))
        for cluster in clusters))


def build_seed_clusters(previous_partition):
    current_ids = set(int(item) for item in common.global_vehicles.keys())
    used_ids = set()
    clusters = []
    for members in previous_partition or []:
        current_members = set(int(item) for item in members) & current_ids
        if current_members:
            clusters.append(common.Cluster(current_members))
            used_ids.update(current_members)
    for vehicle_id in sorted(current_ids - used_ids):
        clusters.append(common.Cluster({vehicle_id}))
    return clusters


def run_one(frame, protocol, seed_partition=None):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=EGO_CAV_ID,
        protocol=protocol,
        density_threshold=RHO_TH)
    algorithm = CountedPotentialVerifiedCOVCoalitionGame(world)
    algorithm.p.N_max = N_MAX
    algorithm.p.T_min_stab = T_MIN_STAB
    algorithm.initialize_vehicles()
    if seed_partition is not None:
        algorithm.coalitions[:] = build_seed_clusters(seed_partition)
    clusters = algorithm.coalition_formation(MAX_ITER)
    return canonical_partition(clusters), algorithm.profile


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    index = int(round((pct / 100.0) * (len(values) - 1)))
    return values[index]


def stats(values):
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p95": percentile(values, 95) if values else 0.0,
        "max": max(values) if values else 0.0,
        "min": min(values) if values else 0.0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = OPV2VFrameDataset(DATASET_ROOT)
    protocol = load_protocol(dataset, SCENARIO_ID)
    timestamps = selected_timestamps(dataset, SCENARIO_ID)
    rows = []
    previous_warm_partition = None
    previous_cold_partition = None
    for frame_index, timestamp in enumerate(timestamps):
        frame = dataset.load_frame(
            SCENARIO_ID,
            timestamp,
            ego_cav_id=EGO_CAV_ID)
        cold_partition, cold_profile = run_one(frame, protocol)
        if frame_index == 0:
            warm_partition, warm_profile = cold_partition, dict(cold_profile)
            warm_mode = "cold_first_frame"
        else:
            warm_partition, warm_profile = run_one(
                frame,
                protocol,
                seed_partition=previous_warm_partition)
            warm_mode = "previous_warm_partition"
        rows.append({
            "frame_index": frame_index + 1,
            "timestamp": timestamp,
            "warm_mode": warm_mode,
            "cold_rounds": cold_profile["rounds"],
            "cold_accepted_migrations": cold_profile["accepted_migrations"],
            "cold_potential_checks": cold_profile["potential_checks"],
            "cold_elapsed_ms": "%.4f" % cold_profile["elapsed_ms"],
            "warm_rounds": warm_profile["rounds"],
            "warm_accepted_migrations": warm_profile["accepted_migrations"],
            "warm_potential_checks": warm_profile["potential_checks"],
            "warm_elapsed_ms": "%.4f" % warm_profile["elapsed_ms"],
            "warm_equals_cold": warm_partition == cold_partition,
            "warm_equals_previous_cold": (
                previous_cold_partition is not None and
                warm_partition == previous_cold_partition),
            "warm_clusters": json.dumps(warm_partition),
            "cold_clusters": json.dumps(cold_partition),
        })
        previous_warm_partition = warm_partition
        previous_cold_partition = cold_partition

    csv_path = os.path.join(OUTPUT_DIR, "warm_start_pv_convergence.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    steady_rows = rows[1:]
    summary = {
        "dataset_root": DATASET_ROOT,
        "scenario_id": SCENARIO_ID,
        "frames": len(rows),
        "parameters": {
            "N_max": N_MAX,
            "rho_th": RHO_TH,
            "T_min_stab_s": T_MIN_STAB,
            "max_iter": MAX_ITER,
        },
        "cold_rounds": stats([int(row["cold_rounds"]) for row in rows]),
        "warm_rounds_all": stats([int(row["warm_rounds"]) for row in rows]),
        "warm_rounds_steady_state": stats(
            [int(row["warm_rounds"]) for row in steady_rows]),
        "warm_accepted_migrations_steady_state": stats(
            [int(row["warm_accepted_migrations"]) for row in steady_rows]),
        "warm_potential_checks_steady_state": stats(
            [int(row["warm_potential_checks"]) for row in steady_rows]),
        "warm_elapsed_ms_steady_state": stats(
            [float(row["warm_elapsed_ms"]) for row in steady_rows]),
        "warm_equals_cold_frames": sum(
            1 for row in rows if str(row["warm_equals_cold"]) == "True"),
        "steady_frames_with_zero_migration": sum(
            1 for row in steady_rows
            if int(row["warm_accepted_migrations"]) == 0),
        "steady_frames_with_one_round": sum(
            1 for row in steady_rows if int(row["warm_rounds"]) == 1),
    }
    summary_path = os.path.join(OUTPUT_DIR, "warm_start_pv_summary.json")
    with open(summary_path, "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    md_path = os.path.join(OUTPUT_DIR, "WARM_START_PV_SUMMARY.md")
    with open(md_path, "w") as stream:
        stream.write("# PV Coalition Warm-Start Convergence\n\n")
        stream.write("- dataset root: `%s`\n" % DATASET_ROOT)
        stream.write("- scenario: `%s`\n" % SCENARIO_ID)
        stream.write("- frames: `%d`\n" % len(rows))
        stream.write("- parameters: `N_max=%d`, `rho_th=%.1f`, "
                     "`T_min_stab=%.1f s`\n\n" %
                     (N_MAX, RHO_TH, T_MIN_STAB))
        stream.write("## Summary\n\n")
        stream.write("| Metric | mean | median | p95 | max | min |\n")
        stream.write("|---|---:|---:|---:|---:|---:|\n")
        for label, key in [
                ("Cold-start rounds", "cold_rounds"),
                ("Warm-start rounds, steady-state",
                 "warm_rounds_steady_state"),
                ("Warm-start accepted migrations, steady-state",
                 "warm_accepted_migrations_steady_state"),
                ("Warm-start potential checks, steady-state",
                 "warm_potential_checks_steady_state"),
                ("Warm-start elapsed ms, steady-state",
                 "warm_elapsed_ms_steady_state")]:
            value = summary[key]
            stream.write("| %s | %.2f | %.2f | %.2f | %.2f | %.2f |\n" %
                         (label, value["mean"], value["median"],
                          value["p95"], value["max"], value["min"]))
        stream.write("\n")
        stream.write("- steady-state frames with one round: `%d/%d`\n" %
                     (summary["steady_frames_with_one_round"],
                      max(0, len(rows) - 1)))
        stream.write("- steady-state frames with zero accepted migration: "
                     "`%d/%d`\n" %
                     (summary["steady_frames_with_zero_migration"],
                      max(0, len(rows) - 1)))
        stream.write("- warm-start final partition equals cold-start final "
                     "partition: `%d/%d` frames\n" %
                     (summary["warm_equals_cold_frames"], len(rows)))
        stream.write("\nRaw per-frame records are in "
                     "`warm_start_pv_convergence.csv`.\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
