import csv
import json
import pathlib
import time

from opencda.tools.offline_inference import (
    OPV2VFrameDataset,
    apply_sgcp_constraint,
    load_protocol,
)
from opencda.tools.offline_ns3_replay import select_timestamps


def main():
    out = pathlib.Path(__file__).resolve().parent
    dataset = OPV2VFrameDataset(r"D:\Data\Carla")
    scenario_id = "2026_07_15_01_26_56"
    protocol = load_protocol(dataset, scenario_id)
    timestamps = select_timestamps(
        dataset,
        scenario_id,
        max_frames=0,
        start_index=0,
        frame_step=1)
    rows = []
    for index, timestamp in enumerate(timestamps, 1):
        frame = dataset.load_frame(scenario_id, timestamp, ego_cav_id=1)
        start = time.time()
        items = apply_sgcp_constraint(
            frame,
            protocol,
            1,
            "fullperception_pcs",
            "all-cavs",
            clustering="singleton",
            num_channels=10,
            bandwidth_mhz=20,
            timestamp=timestamp)
        elapsed = time.time() - start
        scheduled = [
            metadata for _, metadata in items
            if int(metadata.get("communication_bytes", 0)) > 0
        ]
        total_bytes = sum(
            int(metadata.get("communication_bytes", 0))
            for _, metadata in items)
        for _, metadata in items:
            rows.append({
                "timestamp": timestamp,
                "receiver_id": metadata.get("receiver_id"),
                "communication_bytes": metadata.get("communication_bytes"),
                "source_cav_ids": ";".join(
                    map(str, metadata.get("source_cav_ids", []))),
                "uploaded_source_ids": ";".join(
                    map(str, metadata.get("uploaded_source_ids", []))),
                "selected_grid_counts_json": json.dumps(
                    metadata.get("selected_grid_counts", {}),
                    sort_keys=True),
                "channel_allocation": json.dumps(
                    {
                        "%s>%s" % (source, target): channel
                        for (source, target), channel in
                        (metadata.get("channel_allocation") or {}).items()
                    },
                    sort_keys=True),
            })
        print(
            "frame=%s/%s ts=%s receivers=%s scheduled=%s bytes=%s seconds=%.2f"
            % (
                index,
                len(timestamps),
                timestamp,
                len(items),
                len(scheduled),
                total_bytes,
                elapsed),
            flush=True)

    output_path = out / "pcs_singleton_allcavs_metadata_41f.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %s" % output_path)


if __name__ == "__main__":
    main()
