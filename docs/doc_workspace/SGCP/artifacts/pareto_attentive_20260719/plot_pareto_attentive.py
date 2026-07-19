from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "pareto_attentive_source.csv"

CATEGORY_STYLE = {
    "proposed": ("#d62728", "o"),
    "sgcp_ablation": ("#ff7f0e", "o"),
    "scheduler_baseline": ("#1f77b4", "s"),
    "scheduler_baseline_proxy": ("#6baed6", "s"),
    "baseline_protocol_native": ("#9467bd", "^"),
    "edge_assisted_reference": ("#7f7f7f", "X"),
    "prediction_sharing_reference": ("#17becf", "P"),
    "upper_reference": ("#000000", "*"),
    "lower_reference": ("#8c564b", "v"),
}

ANNOTATIONS = {
    "SGCP_PAPG_attentive": "SGCP-PAPG",
    "RandomBudget_attentive": "Random",
    "DensityGreedy_attentive": "Density",
    "EdgeCooperHD_attentive": "EdgeCooper-HD",
    "PACP_LiDAR_attentive": "PACP-LiDAR",
    "Full20Early_attentive": "Full 20-CAV",
    "PureLateBroadcast80_attentive": "Pure late\nbroadcast",
    "PureLateAllToAll80_attentive": "Pure late\nall-to-all",
    "FullPerceptionPCS_attentive": "FullPerception-PCS",
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
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)

    for category, group in df.groupby("category"):
        color, marker = CATEGORY_STYLE.get(category, ("#333333", "o"))
        size = 82 if category != "upper_reference" else 150
        kwargs = {
            "s": size,
            "marker": marker,
            "c": color,
            "label": category.replace("_", " "),
            "alpha": 0.95,
            "linewidths": 0.45,
        }
        if marker != "x":
            kwargs["edgecolors"] = "black"
        ax.scatter(group["total_mbps"], group[metric], **kwargs)

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
            "PureLateBroadcast80_attentive": (2.4, -0.030),
            "PureLateAllToAll80_attentive": (2.0, -0.030),
            "FullPerceptionPCS_attentive": (1.0, 0.018),
            "SGCP_PAPG_attentive": (1.8, 0.020),
            "RandomBudget_attentive": (1.8, -0.035),
            "DensityGreedy_attentive": (2.0, -0.018),
            "EdgeCooperHD_attentive": (1.6, -0.045),
            "PACP_LiDAR_attentive": (2.0, 0.006),
            "Full20Early_attentive": (1.5, 0.012),
        },
        "ap_07": {
            "PureLateBroadcast80_attentive": (2.4, -0.012),
            "PureLateAllToAll80_attentive": (2.0, -0.012),
            "FullPerceptionPCS_attentive": (1.2, 0.012),
            "SGCP_PAPG_attentive": (-11.0, -0.034),
            "RandomBudget_attentive": (1.5, 0.018),
            "DensityGreedy_attentive": (2.0, 0.014),
            "EdgeCooperHD_attentive": (5.0, -0.035),
            "PACP_LiDAR_attentive": (2.0, -0.018),
            "Full20Early_attentive": (1.5, 0.012),
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
            fontsize=7.2,
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#666666"},
        )

    ax.set_xlabel("Communication overhead (Mbps)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-4, 126)
    ax.set_ylim((0.10, 0.92) if metric == "ap_03" else (0.08, 0.50))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="lower right", fontsize=6.6, ncol=2, frameon=True)
    ax.set_title("Attentive SGCP AP-Mbps Pareto Source Points")
    fig.tight_layout()
    fig.savefig(ROOT / f"{out_stem}.png", bbox_inches="tight")
    fig.savefig(ROOT / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA)
    plot_metric(df, "ap_03", "Aggregate AP@0.3", "figure1_pareto_ap03_attentive")
    plot_metric(df, "ap_07", "Aggregate AP@0.7", "figure1_pareto_ap07_attentive")


if __name__ == "__main__":
    main()
