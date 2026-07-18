"""Plot compact BEV case-study panels for SGCP failure diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[5]
DIAG_DIR = (
    ROOT
    / "docs"
    / "doc_workspace"
    / "SGCP"
    / "artifacts"
    / "failure_diag_target_aware_pg_10ch_rho3_41f"
)
OUT_DIR = Path(__file__).resolve().parent

CASES = [
    {
        "timestamp": "000068",
        "object_id": "438",
        "title": "A. Best-view sender missed",
        "best_source": "12",
        "scheduled_source": "9",
        "head": "4",
        "note": "CAV12 has target-grid support; CAV9 was scheduled.",
    },
    {
        "timestamp": "000066",
        "object_id": "401",
        "title": "B. Sparse selected sender",
        "best_source": "4",
        "scheduled_source": "7",
        "head": "12",
        "note": "Same cluster contains useful view, but selected upload is sparse.",
    },
    {
        "timestamp": "000062",
        "object_id": "337",
        "title": "C. Dense grid, weak object support",
        "best_source": "8",
        "scheduled_source": "2",
        "head": "1",
        "note": "Dense grid still needs peer-view object support.",
    },
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (DIAG_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grid_center(grid_id: str) -> tuple[float, float]:
    gx, gy = [int(v) for v in grid_id.split("_")]
    return gx * 10.0 + 5.0, gy * 10.0 + 5.0


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def plot_case(ax, case, gt_rows, vehicles, schedules) -> dict[str, str]:
    ts = case["timestamp"]
    oid = case["object_id"]
    gt = next(r for r in gt_rows if r["timestamp"] == ts and r["object_id"] == oid)
    frame_vehicles = [r for r in vehicles if r["timestamp"] == ts]
    frame_schedules = [r for r in schedules if r["timestamp"] == ts]

    clusters = {}
    for v in frame_vehicles:
        clusters.setdefault(v["cluster_index"], []).append(v)

    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    for idx, (cluster_id, members) in enumerate(sorted(clusters.items(), key=lambda kv: int(kv[0]))):
        color = colors[idx % len(colors)]
        xs = [as_float(v, "x") for v in members]
        ys = [as_float(v, "y") for v in members]
        ax.scatter(xs, ys, s=28, color=color, alpha=0.35, edgecolors="none")
        for v in members:
            cav_id = v["cav_id"]
            is_head = v["is_cluster_head"] == "1"
            marker_size = 95 if is_head else 45
            edge = "#111111" if is_head else color
            face = color if is_head else "white"
            ax.scatter(
                as_float(v, "x"),
                as_float(v, "y"),
                s=marker_size,
                marker="s" if is_head else "o",
                facecolors=face,
                edgecolors=edge,
                linewidths=1.2,
                zorder=4 if is_head else 3,
            )
            if cav_id in {case["best_source"], case["scheduled_source"], case["head"], gt["nearest_cav"]}:
                ax.text(
                    as_float(v, "x") + 0.8,
                    as_float(v, "y") + 0.8,
                    f"C{cav_id}",
                    fontsize=8,
                    color="#111111",
                    zorder=5,
                )

    target_grid = gt["object_grid_id"]
    gcx, gcy = grid_center(target_grid)
    ax.add_patch(
        Rectangle(
            (gcx - 5.0, gcy - 5.0),
            10.0,
            10.0,
            facecolor="#ffd166",
            edgecolor="#8a5a00",
            alpha=0.26,
            linewidth=1.8,
            zorder=1,
        )
    )
    ax.scatter(as_float(gt, "world_x"), as_float(gt, "world_y"), marker="*", s=140, color="#d62828", zorder=6)
    ax.text(as_float(gt, "world_x") + 0.8, as_float(gt, "world_y") + 0.8, f"GT {oid}", fontsize=8, color="#d62828")

    vehicle_by_id = {v["cav_id"]: v for v in frame_vehicles}
    for sched in frame_schedules:
        sender = vehicle_by_id.get(sched["sender_id"])
        receiver = vehicle_by_id.get(sched["receiver_id"])
        if not sender or not receiver:
            continue
        contains_target = target_grid in sched.get("grid_ids_head", "").split(";")
        highlight = (
            sched["sender_id"] == case["scheduled_source"]
            and sched["receiver_id"] == case["head"]
        ) or contains_target
        ax.annotate(
            "",
            xy=(as_float(receiver, "x"), as_float(receiver, "y")),
            xytext=(as_float(sender, "x"), as_float(sender, "y")),
            arrowprops=dict(
                arrowstyle="->",
                color="#d62828" if highlight else "#6c757d",
                linewidth=1.5 if highlight else 0.55,
                alpha=0.82 if highlight else 0.28,
            ),
            zorder=2,
        )

    best = vehicle_by_id.get(case["best_source"])
    if best:
        ax.scatter(as_float(best, "x"), as_float(best, "y"), s=150, facecolors="none", edgecolors="#2a9d8f", linewidths=2.2, zorder=7)
    head = vehicle_by_id.get(case["head"])
    if head:
        ax.scatter(as_float(head, "x"), as_float(head, "y"), s=170, facecolors="none", edgecolors="#111111", linewidths=2.0, zorder=7)

    ax.set_title(f'{case["title"]}\nframe {ts}, object {oid}, grid {target_grid}', fontsize=9)
    ax.text(0.01, 0.02, case["note"], transform=ax.transAxes, fontsize=8, va="bottom")
    focus_x = [gcx, as_float(gt, "world_x")]
    focus_y = [gcy, as_float(gt, "world_y")]
    for cav_id in {case["best_source"], case["scheduled_source"], case["head"], gt["nearest_cav"]}:
        cav = vehicle_by_id.get(cav_id)
        if cav:
            focus_x.append(as_float(cav, "x"))
            focus_y.append(as_float(cav, "y"))
    ax.set_xlim(min(gcx - 45, min(focus_x) - 10), max(gcx + 45, max(focus_x) + 10))
    ax.set_ylim(min(gcy - 45, min(focus_y) - 10), max(gcy + 45, max(focus_y) + 10))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")

    return {
        "timestamp": ts,
        "object_id": oid,
        "target_grid": target_grid,
        "nearest_cav": gt["nearest_cav"],
        "nearest_head": gt["nearest_head"],
        "best_source": case["best_source"],
        "scheduled_source": case["scheduled_source"],
        "full_reference_matched": gt["full_reference_matched"],
        "method_matched": gt["method_matched"],
        "scheduled_covering_links": gt["scheduled_covering_links"],
        "nearest_cav_object_grid_points": gt["nearest_cav_object_grid_points"],
        "nearest_head_covering_point_count": gt["nearest_head_covering_point_count"],
    }


def main() -> None:
    gt_rows = read_csv("gt_objects.csv")
    vehicles = read_csv("vehicles.csv")
    schedules = read_csv("schedules.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), constrained_layout=True)
    summary = [plot_case(ax, case, gt_rows, vehicles, schedules) for ax, case in zip(axes, CASES)]
    fig.suptitle("SGCP qualitative failure cases under 20 MHz / 10 subchannels", fontsize=13)

    png = OUT_DIR / "qualitative_case_study_bev.png"
    pdf = OUT_DIR / "qualitative_case_study_bev.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    with (OUT_DIR / "qualitative_case_study_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
