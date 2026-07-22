import argparse
import csv
import json
import os


FIELDNAMES = [
    "timestamp",
    "area_id",
    "source_id",
    "target_id",
    "bytes",
    "upload_type",
    "pkt_id",
    "sc_start",
    "sc_num",
    "slot_index",
    "stage",
    "scheduled_delay_ms",
]


def parse_channel_allocation(raw):
    mapping = {}
    for item in (raw or "").split(";"):
        item = item.strip()
        if not item or ">" not in item or ":" not in item:
            continue
        pair, channel = item.split(":", 1)
        source, target = pair.split(">", 1)
        mapping[(int(source), int(target))] = int(channel)
    return mapping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--timestamp", default="000060")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=10000)
    parser.add_argument("--point-bytes", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    pkt_id = 1
    with open(args.trace, newline="") as stream:
        for trace_row in csv.DictReader(stream):
            if trace_row["timestamp"] != args.timestamp:
                continue
            receiver_id = int(trace_row["receiver_id"])
            point_counts = json.loads(
                trace_row.get("point_counts_json") or "{}")
            uploaded_sources = [
                int(item)
                for item in (trace_row.get("uploaded_source_ids") or "").split(";")
                if item
            ]
            channel_by_link = parse_channel_allocation(
                trace_row.get("channel_allocation"))
            for source_id in uploaded_sources:
                payload_bytes = int(point_counts.get(str(source_id), 0)) * int(
                    args.point_bytes)
                sc_start = channel_by_link.get((source_id, receiver_id), 0)
                remaining = payload_bytes
                while remaining > 0:
                    chunk = min(args.chunk_bytes, remaining)
                    rows.append({
                        "timestamp": args.timestamp,
                        "area_id": "",
                        "source_id": source_id,
                        "target_id": receiver_id,
                        "bytes": chunk,
                        "upload_type": "sgcp_papg_200ms_exact",
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
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print("chunks=%d bytes=%d links=%d" % (
        len(rows),
        sum(int(row["bytes"]) for row in rows),
        len({(row["source_id"], row["target_id"]) for row in rows})))


if __name__ == "__main__":
    main()
