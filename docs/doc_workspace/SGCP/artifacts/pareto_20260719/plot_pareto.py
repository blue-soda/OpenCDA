from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "pareto_source.csv"


CATEGORY_STYLE = {
    "proposed": ("#d62728", "o"),
    "sgcp_ablation": ("#ff7f0e", "o"),
    "sgcp_sensitivity": ("#ffbb78", "o"),
    "scheduler_baseline": ("#1f77b4", "s"),
    "scheduler_baseline_proxy": ("#6baed6", "s"),
    "v2v_proxy_baseline": ("#2ca02c", "D"),
    "baseline_protocol_native": ("#9467bd", "^"),
    "edge_assisted_reference": ("#7f7f7f", "X"),
    "prediction_sharing_reference": ("#17becf", "P"),
    "upper_reference": ("#000000", "*"),
    "lower_reference": ("#8c564b", "v"),
    "negative_probe": ("#bcbd22", "x"),
}


ANNOTATIONS = {
    "SGCP_PAPG": "SGCP-PAPG",
    "RandomBudget": "Random",
    "DensityGreedy": "Density",
    "EdgeCooperHD": "EdgeCooper-HD",
    "PACP_LiDAR": "PACP-LiDAR",
    "Full20Early": "Full 20-CAV",
    "PureLateBroadcast80": "Pure late\nbroadcast",
    "PureLateAllToAll80": "Pure late\nall-to-all",
    "FullPerceptionPCS": "FullPerception-PCS",
}


def frontier(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    best = -1.0
    picked = []
    for _, row in rows.sort_values("total_mbps").iterrows():
        value = float(row[metric])
        if value > best + 1e-9:
            picked.append(row)
            best = value
    return pd.DataFrame(picked)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)

    for category, group in df.groupby("category"):
        color, marker = CATEGORY_STYLE.get(category, ("#333333", "o"))
        alpha = 0.95
        size = 80 if category != "upper_reference" else 150
        scatter_kwargs = {
            "s": size,
            "marker": marker,
            "c": color,
            "label": category.replace("_", " "),
            "alpha": alpha,
            "linewidths": 0.45,
        }
        if marker != "x":
            scatter_kwargs["edgecolors"] = "black"
        ax.scatter(group["total_mbps"], group[metric], **scatter_kwargs)

    raw = df[df["category"] != "prediction_sharing_reference"].copy()
    raw_frontier = frontier(raw, metric)
    ax.plot(
        raw_frontier["total_mbps"],
        raw_frontier[metric],
        color="#444444",
        linewidth=1.2,
        linestyle="--",
        label="raw-LiDAR frontier",
    )

    offsets = {
        "ap_03": {
            "PureLateBroadcast80": (2.5, -0.030),
            "PureLateAllToAll80": (2.0, -0.030),
            "FullPerceptionPCS": (1.0, 0.015),
            "SGCP_PAPG": (1.8, 0.022),
            "RandomBudget": (1.8, -0.028),
            "DensityGreedy": (2.2, -0.018),
            "EdgeCooperHD": (2.0, 0.018),
            "PACP_LiDAR": (2.0, 0.006),
            "Full20Early": (1.5, 0.012),
        },
        "ap_07": {
            "PureLateBroadcast80": (2.5, -0.012),
            "PureLateAllToAll80": (2.0, -0.012),
            "FullPerceptionPCS": (1.2, 0.012),
            "SGCP_PAPG": (1.5, 0.018),
            "RandomBudget": (1.5, -0.020),
            "DensityGreedy": (2.0, -0.010),
            "EdgeCooperHD": (1.8, 0.018),
            "PACP_LiDAR": (2.0, 0.006),
            "Full20Early": (1.5, 0.012),
        },
    }

    for _, row in df.iterrows():
        label = ANNOTATIONS.get(row["method"])
        if not label:
            continue
        dx, dy = offsets[metric].get(row["method"], (1.4, 0.006))
        ax.annotate(
            label,
            (row["total_mbps"], row[metric]),
            xytext=(row["total_mbps"] + dx, row[metric] + dy),
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#666666"},
        )

    ax.set_xlabel("Communication overhead (Mbps)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-4, 126)
    ax.set_ylim((0.20, 0.88) if metric == "ap_03" else (0.05, 0.55))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="lower right", fontsize=6.8, ncol=2, frameon=True)
    ax.set_title("SGCP AP-Mbps Pareto Source Points")
    fig.tight_layout()
    fig.savefig(ROOT / f"{out_stem}.png", bbox_inches="tight")
    fig.savefig(ROOT / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA)
    plot_metric(df, "ap_03", "Aggregate AP@0.3", "figure1_pareto_ap03")
    plot_metric(df, "ap_07", "Aggregate AP@0.7", "figure1_pareto_ap07")


if __name__ == "__main__":
    main()
