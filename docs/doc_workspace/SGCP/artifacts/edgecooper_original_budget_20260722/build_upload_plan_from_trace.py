import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--timestamp", default="000060")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    pkt_id = 1
    next_channel = 0
    with open(args.trace, newline="") as stream:
        for trace_row in csv.DictReader(stream):
            if trace_row["timestamp"] != args.timestamp:
                continue
            point_counts = json.loads(trace_row.get("point_counts_json") or "{}")
            uploaded_sources = [
                int(item)
                for item in (trace_row.get("uploaded_source_ids") or "").split(";")
                if item
            ]
            receiver_id = int(trace_row["receiver_id"])
            for source_id in uploaded_sources:
                payload_bytes = int(point_counts.get(str(source_id), 0)) * 16
                remaining = payload_bytes
                sc_start = next_channel % 10
                next_channel += 1
                while remaining > 0:
                    chunk = min(args.chunk_bytes, remaining)
                    rows.append({
                        "timestamp": args.timestamp,
                        "area_id": "",
                        "source_id": source_id,
                        "target_id": receiver_id,
                        "bytes": chunk,
                        "upload_type": "edgecooper_trace",
                        "pkt_id": pkt_id,
                        "sc_start": sc_start,
                        "sc_num": 1,
                        "slot_index": "",
                        "stage": "",
                        "scheduled_delay_ms": "",
                    })
                    pkt_id += 1
                    remaining -= chunk
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "timestamp", "area_id", "source_id", "target_id", "bytes",
        "upload_type", "pkt_id", "sc_start", "sc_num", "slot_index",
        "stage", "scheduled_delay_ms",
    ]
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("chunks=%d bytes=%d" % (
        len(rows),
        sum(int(row["bytes"]) for row in rows)))


if __name__ == "__main__":
    main()
