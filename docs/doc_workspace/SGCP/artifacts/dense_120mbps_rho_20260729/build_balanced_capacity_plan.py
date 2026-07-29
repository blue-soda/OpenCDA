# -*- coding: utf-8 -*-
"""Build balanced endpoint-disjoint NS3 capacity upload plans.

The plan uses real CAV positions from the dense CARLA dump, greedily pairs
nearby vehicles into endpoint-disjoint V2V links, and maps the links to
orthogonal subchannels.  It is a channel-capacity probe, not an SGCP scheduler
output.
"""

import argparse
import csv
import math
import pathlib

from opencda.core.common.offline_dataset import OPV2VFrameDataset


DEFAULT_DATASET_ROOT = pathlib.Path(r"D:\Data\Carla")
DEFAULT_SCENARIO_ID = "2026_07_29_02_32_08"
DEFAULT_TIMESTAMP = "000076"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    parser.add_argument("--bytes-per-link", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=10000)
    parser.add_argument("--target-subchannels", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def cav_position(cav_content):
    params = cav_content["params"]
    pose = (
        params.get("true_ego_pos") or
        params.get("predicted_ego_pos") or
        params["lidar_pose"])
    return float(pose[0]), float(pose[1]), float(pose[2])


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2)


def greedy_endpoint_disjoint_pairs(frame, count):
    positions = {
        int(cav_id): cav_position(cav_content)
        for cav_id, cav_content in frame.items()
    }
    candidates = []
    ids = sorted(positions)
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            candidates.append((distance(positions[left], positions[right]),
                               left, right))
    candidates.sort()

    used = set()
    pairs = []
    for dist, left, right in candidates:
        if left in used or right in used:
            continue
        # Alternate direction deterministically so source/receiver ids are
        # not all biased toward the smaller CAV id.
        source, target = (right, left) if len(pairs) % 2 == 0 else (left, right)
        pairs.append((source, target, dist))
        used.add(left)
        used.add(right)
        if len(pairs) >= count:
            break
    if len(pairs) < count:
        raise RuntimeError("Only built %d endpoint-disjoint pairs" % len(pairs))
    return pairs


def write_plan(path, timestamp, pairs, bytes_per_link, chunk_bytes):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pkt_id = 1
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "timestamp", "area_id", "source_id", "target_id", "bytes",
            "upload_type", "pkt_id", "sc_start", "sc_num",
            "slot_index", "stage", "scheduled_delay_ms"])
        writer.writeheader()
        for sc_start, (source, target, _) in enumerate(pairs):
            remaining = int(bytes_per_link)
            while remaining > 0:
                size = min(int(chunk_bytes), remaining)
                writer.writerow({
                    "timestamp": timestamp,
                    "area_id": "",
                    "source_id": source,
                    "target_id": target,
                    "bytes": size,
                    "upload_type": "balanced_capacity",
                    "pkt_id": pkt_id,
                    "sc_start": sc_start,
                    "sc_num": 1,
                    "slot_index": "",
                    "stage": "",
                    "scheduled_delay_ms": "",
                })
                pkt_id += 1
                remaining -= size


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)
    frame = dataset.load_frame(
        args.scenario_id,
        args.timestamp,
        add_transformation=False)
    pairs = greedy_endpoint_disjoint_pairs(
        frame,
        int(args.target_subchannels))
    write_plan(
        args.output,
        args.timestamp,
        pairs,
        args.bytes_per_link,
        args.chunk_bytes)
    print("links=%d bytes_per_link=%d total_bytes=%d logical_mbps=%.3f" % (
        len(pairs),
        args.bytes_per_link,
        len(pairs) * args.bytes_per_link,
        len(pairs) * args.bytes_per_link * 8.0 / 1e5))
    for index, (source, target, dist) in enumerate(pairs):
        print("sc=%d %s->%s distance_m=%.2f" % (
            index,
            source,
            target,
            dist))


if __name__ == "__main__":
    main()
