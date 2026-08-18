# -*- coding: utf-8 -*-
"""Profile dense SGCP coalition and scheduler realtime costs.

Artifact-local script for the 2026-07-29 dense SGCP clean package.  It avoids
detector inference and only profiles the distributed coalition maintenance and
dynamic C/V resource-scheduler solving path.
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
from opencda.core.clustering.algorithms.resource_allocation.builder import (
    build_resource_allocator,
)
from opencda.core.clustering.utils import common
from opencda.tools.offline_replay import apply_cluster_state


OUTPUT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DATASET_ROOT = r"D:\Data\Carla"
SCENARIO_ID = "2026_07_29_02_32_08"
EGO_CAV_ID = 1
FRAME_START = 60
FRAME_END = 140
FRAME_STEP = 2
N_MAX = 4
RHO_TH = 2.0
T_MIN_STAB = 0.1
MAX_ITER = 20
RESOURCE_ALLOCATION = "dynamic_cv"
NUM_CHANNELS = 10
BANDWIDTH_MHZ = 40.0
HEAD_RB_BUDGET = 2


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


class ProfiledPVGame(PotentialVerifiedCOVCoalitionGame):
    """Current PV coalition game with cached pairwise values and timers."""

    def __init__(self, cav_world):
        super(ProfiledPVGame, self).__init__(cav_world)
        self._pair_cache = {}
        self.profile = {}

    def build_pair_cache(self):
        start = time.perf_counter()
        vehicle_ids = sorted(int(item) for item in common.global_vehicles)
        cache = {}
        for index, first_id in enumerate(vehicle_ids):
            for second_id in vehicle_ids[index + 1:]:
                key = (first_id, second_id)
                cache[key] = super(ProfiledPVGame, self)._pairwise_view_value(
                    first_id, second_id)
        self._pair_cache = cache
        return (time.perf_counter() - start) * 1000.0

    def _pairwise_view_value(self, first_id, second_id):
        first_id = int(first_id)
        second_id = int(second_id)
        if first_id == second_id:
            return 0.0
        key = tuple(sorted((first_id, second_id)))
        if key not in self._pair_cache:
            self._pair_cache[key] = super(
                ProfiledPVGame, self)._pairwise_view_value(first_id, second_id)
        return self._pair_cache[key]

    def coalition_formation(self, max_iter=20):
        self.check_is_ok()
        self.ego_coalition_be_first()
        self.capacity_stats = {
            "full_candidate_skips": 0,
        }
        self.last_potential_checks = []
        pair_cache_ms = self.build_pair_cache()
        vehicle_times = {
            int(vehicle_id): 0.0 for vehicle_id in common.global_vehicles
        }
        rounds = 0
        accepted_migrations = 0
        potential_checks = 0
        proxy_accepts = 0
        total_start = time.perf_counter()
        for iteration in range(max_iter):
            rounds = iteration + 1
            updated = False
            for vid in list(common.global_vehicles.keys()):
                vid_int = int(vid)
                action_start = time.perf_counter()
                current = self.find_coalition(vid)
                if current is None:
                    vehicle_times[vid_int] += (
                        time.perf_counter() - action_start) * 1000.0
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
                    if not proxy_accept or phi_delta <= 1e-9:
                        continue
                    candidate = {
                        "coalition": coalition,
                        "proxy_after": delta,
                        "phi_delta": phi_delta,
                    }
                    if best_candidate is None:
                        best_candidate = candidate
                    elif (delta, phi_delta) > (
                            best_candidate["proxy_after"],
                            best_candidate["phi_delta"]):
                        best_candidate = candidate

                if best_candidate is not None:
                    best_coalition = best_candidate["coalition"]
                    current.remove_member(vid)
                    best_coalition.add_member(vid)
                    accepted_migrations += 1
                    if current.size() == 0:
                        self.coalitions.remove(current)
                        current = None
                    if current is not None and current.head_id in current.members:
                        current.grid_bits = current.compute_grid_bits()
                    if best_coalition.head_id in best_coalition.members:
                        best_coalition.grid_bits = (
                            best_coalition.compute_grid_bits())
                    updated = True
                vehicle_times[vid_int] += (
                    time.perf_counter() - action_start) * 1000.0
            if not updated:
                break
        total_ms = (time.perf_counter() - total_start) * 1000.0
        self.profile = {
            "rounds": rounds,
            "accepted_migrations": accepted_migrations,
            "potential_checks": potential_checks,
            "proxy_accepts": proxy_accepts,
            "pair_cache_ms": pair_cache_ms,
            "serial_admission_ms": total_ms,
            "distributed_admission_ms": max(vehicle_times.values())
            if vehicle_times else 0.0,
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


def run_coalition(frame, protocol, seed_partition=None):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=EGO_CAV_ID,
        protocol=protocol,
        density_threshold=RHO_TH)
    algorithm = ProfiledPVGame(world)
    algorithm.p.N_max = N_MAX
    algorithm.p.T_min_stab = T_MIN_STAB
    algorithm.initialize_vehicles()
    if seed_partition is not None:
        algorithm.coalitions[:] = build_seed_clusters(seed_partition)
    clusters = algorithm.coalition_formation(MAX_ITER)
    return world, clusters, canonical_partition(clusters), dict(algorithm.profile)


def run_scheduler(world, clusters):
    apply_cluster_state(world, clusters)
    scheduler = build_resource_allocator(RESOURCE_ALLOCATION, world)
    if hasattr(scheduler, "p"):
        scheduler.p.num_channels = NUM_CHANNELS
        scheduler.p.bandwidth_all = BANDWIDTH_MHZ * 1e6
        scheduler.p.head_rb_budget = HEAD_RB_BUDGET
    scheduler.set_clusters(clusters)
    start = time.perf_counter()
    scheduler.run()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    stats_dict = getattr(scheduler, "convergence_stats", {}) or {}
    return elapsed_ms, stats_dict


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = OPV2VFrameDataset(DATASET_ROOT)
    protocol = load_protocol(dataset, SCENARIO_ID)
    timestamps = selected_timestamps(dataset, SCENARIO_ID)
    rows = []
    previous_warm_partition = None
    for frame_index, timestamp in enumerate(timestamps):
        frame = dataset.load_frame(
            SCENARIO_ID,
            timestamp,
            ego_cav_id=EGO_CAV_ID)
        cold_world, cold_clusters, cold_partition, cold_profile = (
            run_coalition(frame, protocol))
        scheduler_ms, scheduler_stats = run_scheduler(cold_world, cold_clusters)
        if frame_index == 0:
            warm_partition = cold_partition
            warm_profile = dict(cold_profile)
            warm_mode = "cold_first_frame"
        else:
            _, _, warm_partition, warm_profile = run_coalition(
                frame,
                protocol,
                seed_partition=previous_warm_partition)
            warm_mode = "previous_warm_partition"
        previous_warm_partition = warm_partition
        rows.append({
            "frame_index": frame_index + 1,
            "timestamp": timestamp,
            "warm_mode": warm_mode,
            "cold_rounds": cold_profile["rounds"],
            "cold_accepted_migrations": cold_profile["accepted_migrations"],
            "cold_potential_checks": cold_profile["potential_checks"],
            "cold_pair_cache_ms": "%.4f" % cold_profile["pair_cache_ms"],
            "cold_serial_admission_ms":
                "%.4f" % cold_profile["serial_admission_ms"],
            "cold_distributed_admission_ms":
                "%.4f" % cold_profile["distributed_admission_ms"],
            "warm_rounds": warm_profile["rounds"],
            "warm_accepted_migrations": warm_profile["accepted_migrations"],
            "warm_potential_checks": warm_profile["potential_checks"],
            "warm_pair_cache_ms": "%.4f" % warm_profile["pair_cache_ms"],
            "warm_serial_admission_ms":
                "%.4f" % warm_profile["serial_admission_ms"],
            "warm_distributed_admission_ms":
                "%.4f" % warm_profile["distributed_admission_ms"],
            "scheduler_ms": "%.4f" % scheduler_ms,
            "scheduler_iterations": int(scheduler_stats.get("iterations", 0)),
            "scheduler_links": int(scheduler_stats.get("scheduled_links", 0)),
            "scheduler_selected_grids":
                int(scheduler_stats.get("selected_grids", 0)),
            "scheduler_used_rbs": int(scheduler_stats.get("used_rbs", 0)),
            "cold_clusters": json.dumps(cold_partition),
            "warm_clusters": json.dumps(warm_partition),
        })

    csv_path = os.path.join(OUTPUT_DIR, "realtime_profile_dense.csv")
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
            "resource_allocation": RESOURCE_ALLOCATION,
            "num_channels": NUM_CHANNELS,
            "bandwidth_mhz": BANDWIDTH_MHZ,
            "head_rb_budget": HEAD_RB_BUDGET,
        },
        "cold_rounds": stats([int(row["cold_rounds"]) for row in rows]),
        "cold_accepted_migrations":
            stats([int(row["cold_accepted_migrations"]) for row in rows]),
        "cold_potential_checks":
            stats([int(row["cold_potential_checks"]) for row in rows]),
        "cold_pair_cache_ms":
            stats([float(row["cold_pair_cache_ms"]) for row in rows]),
        "cold_distributed_admission_ms": stats(
            [float(row["cold_distributed_admission_ms"]) for row in rows]),
        "warm_rounds_steady_state":
            stats([int(row["warm_rounds"]) for row in steady_rows]),
        "warm_accepted_migrations_steady_state":
            stats([int(row["warm_accepted_migrations"]) for row in steady_rows]),
        "warm_potential_checks_steady_state":
            stats([int(row["warm_potential_checks"]) for row in steady_rows]),
        "warm_distributed_admission_ms_steady_state": stats(
            [float(row["warm_distributed_admission_ms"])
             for row in steady_rows]),
        "scheduler_ms":
            stats([float(row["scheduler_ms"]) for row in rows]),
        "scheduler_links":
            stats([int(row["scheduler_links"]) for row in rows]),
        "scheduler_selected_grids":
            stats([int(row["scheduler_selected_grids"]) for row in rows]),
        "scheduler_used_rbs":
            stats([int(row["scheduler_used_rbs"]) for row in rows]),
    }
    summary_path = os.path.join(OUTPUT_DIR, "realtime_profile_dense_summary.json")
    with open(summary_path, "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
