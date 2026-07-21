import csv
import json
import os
import statistics
import argparse

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.tools.offline_inference import (
    apply_sgcp_constraint,
    load_protocol,
)


DATASET_ROOT = r"D:\Data\Carla"
SCENARIO_ID = "2026_07_15_01_26_56"
OUTPUT_DIR = r"docs\doc_workspace\SGCP\artifacts\pcs_rounds_20260721"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile PCS repeated-round deadline admission.")
    parser.add_argument("--max-frames", type=int, default=11,
                        help="Maximum frames to profile. Use 0 for all.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trace_out = os.path.join(
        OUTPUT_DIR,
        "pcs_div4_r6_d60_41f_metadata_trace.csv")
    summary_out = os.path.join(
        OUTPUT_DIR,
        "pcs_div4_r6_d60_41f_metadata_summary.csv")

    dataset = OPV2VFrameDataset(DATASET_ROOT)
    protocol = load_protocol(dataset, SCENARIO_ID)
    frames = dataset.scenarios[SCENARIO_ID]["timestamps"]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    rows = []
    frame_summaries = []
    for index, timestamp in enumerate(frames, start=1):
        frame = dataset.load_frame(SCENARIO_ID, timestamp)
        items = apply_sgcp_constraint(
            frame,
            protocol,
            "1",
            "fullperception_pcs",
            "all-scheduled-receivers",
            clustering="singleton",
            num_channels=10,
            bandwidth_mhz=20,
            pcs_frame_rounds=6,
            pcs_frame_deadline_ms=60)
        frame_bytes = 0
        frame_time = 0.0
        rounds = 0
        for _, metadata in items:
            frame_bytes += int(metadata.get("communication_bytes") or 0)
            frame_time = max(
                frame_time,
                float(metadata.get("frame_comm_time_ms") or 0))
            rounds = max(
                rounds,
                int(metadata.get("pcs_rounds_accepted") or 0))
            rows.append({
                "timestamp": timestamp,
                "receiver_id": metadata.get("receiver_id"),
                "communication_bytes": metadata.get("communication_bytes"),
                "frame_comm_time_ms": metadata.get("frame_comm_time_ms"),
                "pcs_rounds_requested": metadata.get(
                    "pcs_rounds_requested"),
                "pcs_rounds_accepted": metadata.get("pcs_rounds_accepted"),
                "pcs_round_comm_time_ms_json": json.dumps(
                    metadata.get("pcs_round_comm_time_ms", [])),
                "pcs_round_comm_bytes_json": json.dumps(
                    metadata.get("pcs_round_comm_bytes", [])),
                "source_cav_ids": ";".join(
                    str(item) for item in metadata.get(
                        "source_cav_ids", [])),
                "selected_grid_counts_json": json.dumps(
                    metadata.get("selected_grid_counts", {}),
                    sort_keys=True),
            })
        frame_summaries.append({
            "timestamp": timestamp,
            "receiver_rows": len(items),
            "frame_bytes": frame_bytes,
            "frame_comm_time_ms": frame_time,
            "rounds_accepted": rounds,
        })
        print(
            index,
            timestamp,
            "receivers",
            len(items),
            "bytes",
            frame_bytes,
            "time_ms",
            frame_time,
            "rounds",
            rounds)

    with open(trace_out, "w", newline="") as stream:
        fieldnames = list(rows[0].keys()) if rows else ["timestamp"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frame_times = [row["frame_comm_time_ms"] for row in frame_summaries]
    frame_bytes = [row["frame_bytes"] for row in frame_summaries]
    summary = {
        "frames": len(frame_summaries),
        "receiver_rows": len(rows),
        "total_bytes": sum(frame_bytes),
        "generated_mbps": (
            sum(frame_bytes) * 8.0 /
            (len(frame_summaries) * 0.1) / 1e6),
        "avg_frame_comm_time_ms": statistics.mean(frame_times),
        "max_frame_comm_time_ms": max(frame_times),
        "min_frame_comm_time_ms": min(frame_times),
        "p95_frame_comm_time_ms": sorted(frame_times)[
            int(0.95 * (len(frame_times) - 1))],
        "avg_frame_bytes": statistics.mean(frame_bytes),
        "avg_rounds_accepted": statistics.mean(
            [row["rounds_accepted"] for row in frame_summaries]),
    }
    with open(summary_out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print("SUMMARY", summary)
    print("wrote", trace_out, summary_out)


if __name__ == "__main__":
    main()
