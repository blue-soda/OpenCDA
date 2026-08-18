import csv
import json
import os


TRACE = os.path.join(os.path.dirname(__file__), "trace.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "sgcp_hybrid_ns3_upload_plan.csv")

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


def main():
    rows = []
    pkt_id = 1
    with open(TRACE, newline="") as stream:
        for trace_row in csv.DictReader(stream):
            receiver_id = int(trace_row["receiver_id"])
            timestamp = trace_row["timestamp"]
            point_counts = json.loads(trace_row.get("point_counts_json") or "{}")
            uploaded_sources = [
                int(item)
                for item in (trace_row.get("uploaded_source_ids") or "").split(";")
                if item
            ]
            channel_by_link = parse_channel_allocation(
                trace_row.get("channel_allocation"))
            for source_id in uploaded_sources:
                payload_bytes = int(point_counts.get(str(source_id), 0)) * 16
                if payload_bytes <= 0:
                    continue
                rows.append({
                    "timestamp": timestamp,
                    "area_id": "",
                    "source_id": source_id,
                    "target_id": receiver_id,
                    "bytes": payload_bytes,
                    "upload_type": "sgcp_hybrid_table1_trace",
                    "pkt_id": pkt_id,
                    "sc_start": channel_by_link.get((source_id, receiver_id), 0),
                    "sc_num": 1,
                    "slot_index": "",
                    "stage": "raw_lidar",
                    "scheduled_delay_ms": "",
                })
                pkt_id += 1

    with open(OUTPUT, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print("rows=%d bytes=%d frames=%d links=%d output=%s" % (
        len(rows),
        sum(int(row["bytes"]) for row in rows),
        len({row["timestamp"] for row in rows}),
        len({(row["source_id"], row["target_id"]) for row in rows}),
        OUTPUT))


if __name__ == "__main__":
    main()
