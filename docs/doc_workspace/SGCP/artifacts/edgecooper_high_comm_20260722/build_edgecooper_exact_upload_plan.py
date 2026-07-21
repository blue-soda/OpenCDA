import csv
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
TRACE = ROOT / "edgecooper_global_exact_d60_r35_m3g200_41f_trace.csv"
OUT = ROOT / "edgecooper_exact_d60_ns3_frame000060" / "upload_plan.csv"


def main():
    rows = []
    pkt_id = 1
    link_to_channel = {}
    with TRACE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["timestamp"] != "000060":
                continue
            point_counts = json.loads(row["point_counts_json"] or "{}")
            for source_id in [
                item for item in (row["uploaded_source_ids"] or "").split(";")
                if item
            ]:
                target_id = row["receiver_id"]
                link = (int(source_id), int(target_id))
                if link not in link_to_channel:
                    link_to_channel[link] = len(link_to_channel) % 10
                payload_bytes = int(point_counts.get(source_id, 0)) * 16
                remaining = payload_bytes
                while remaining > 0:
                    chunk = min(10000, remaining)
                    rows.append({
                        "timestamp": row["timestamp"],
                        "area_id": "",
                        "source_id": link[0],
                        "target_id": link[1],
                        "bytes": chunk,
                        "upload_type": "edgecooper_exact_d60",
                        "pkt_id": pkt_id,
                        "sc_start": link_to_channel[link],
                        "sc_num": 1,
                        "slot_index": "",
                        "stage": "",
                        "scheduled_delay_ms": "",
                    })
                    pkt_id += 1
                    remaining -= chunk

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    print(
        "rows=%s links=%s bytes=%s" %
        (len(rows), len(link_to_channel), total_bytes))


if __name__ == "__main__":
    main()
