import csv
import json
import sys
from pathlib import Path


def safe_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def selected_grid_count(row):
    raw = row.get("selected_grid_counts_json") or "{}"
    try:
        counts = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    return sum(safe_int(value) for value in counts.values())


def summarize(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    payload = sum(safe_int(row.get("communication_bytes")) for row in rows)
    grid_counts = [selected_grid_count(row) for row in rows]
    nonzero = [count for count in grid_counts if count > 0]
    return {
        "trace": str(path),
        "rows": len(rows),
        "payload_bytes": payload,
        "avg_grids_per_row": (
            sum(grid_counts) / float(len(grid_counts)) if grid_counts else 0.0),
        "avg_grids_per_nonzero_row": (
            sum(nonzero) / float(len(nonzero)) if nonzero else 0.0),
        "max_grids_per_row": max(grid_counts) if grid_counts else 0,
    }


def main():
    for path in sys.argv[1:]:
        summary = summarize(path)
        print(",".join([
            Path(path).name,
            str(summary["rows"]),
            str(summary["payload_bytes"]),
            "%.2f" % summary["avg_grids_per_row"],
            "%.2f" % summary["avg_grids_per_nonzero_row"],
            str(summary["max_grids_per_row"]),
        ]))


if __name__ == "__main__":
    main()
