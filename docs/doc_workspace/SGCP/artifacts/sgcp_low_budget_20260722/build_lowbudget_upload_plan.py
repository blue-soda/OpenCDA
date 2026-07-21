import csv
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
TRACE = ROOT / "sgcp_papg_40mhz_10ch_bh2_cap4000_41f_trace.csv"
OUT = ROOT / "ns3_frame000060" / "upload_plan.csv"


def main():
    rows = []
    pkt_id = 1
    with TRACE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["timestamp"] != "000060":
                continue
            point_counts = json.loads(row["point_counts_json"] or "{}")
            channel_allocation = {}
            for item in (row["channel_allocation"] or "").split(";"):
                if not item:
                    continue
                link, channel = item.split(":")
                source_id, target_id = link.split(">")
                channel_allocation[(source_id, target_id)] = int(channel)

            for source_id in [
                item for item in (row["uploaded_source_ids"] or "").split(";")
                if item
            ]:
                target_id = row["receiver_id"]
                payload_bytes = int(point_counts.get(source_id, 0)) * 16
                if payload_bytes <= 0:
                    continue
                sc_start = channel_allocation.get((source_id, target_id), 0)
                remaining = payload_bytes
                while remaining > 0:
                    chunk = min(10000, remaining)
                    rows.append({
                        "timestamp": row["timestamp"],
                        "area_id": "",
                        "source_id": int(source_id),
                        "target_id": int(target_id),
                        "bytes": chunk,
                        "upload_type": "sgcp_papg_lowbudget_cap4000",
                        "pkt_id": pkt_id,
                        "sc_start": sc_start,
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
    link_count = len({(row["source_id"], row["target_id"]) for row in rows})
    print("rows=%s links=%s bytes=%s" % (len(rows), link_count, total_bytes))


if __name__ == "__main__":
    main()
