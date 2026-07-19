from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SGCP_ROOT = ROOT.parents[1]

PROTOCOL_CSV = (
    SGCP_ROOT
    / "artifacts"
    / "attentive_protocol_20260719"
    / "protocol_native_attentive_manifest.csv"
)
FUSION_CSV = (
    SGCP_ROOT
    / "artifacts"
    / "attentive_fusion_ablation_20260719"
    / "fusion_scaffold_attentive_manifest.csv"
)
SCHEDULER_CSV = (
    SGCP_ROOT
    / "artifacts"
    / "attentive_scheduler_comparison_20260719"
    / "scheduler_comparison_attentive_manifest.csv"
)

AP_COLS = ["ap_03", "ap_05", "ap_07"]
AP_LABELS = ["AP@0.3", "AP@0.5", "AP@0.7"]
COLORS = ["#4c78a8", "#f58518", "#54a24b"]


def grouped_bar(
    df: pd.DataFrame,
    labels: List[str],
    title: str,
    out_stem: str,
    comm_labels: List[str],
    ylim: float = 0.94,
) -> None:
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=180)

    for i, (col, label, color) in enumerate(zip(AP_COLS, AP_LABELS, COLORS)):
        ax.bar(
            x + (i - 1) * width,
            df[col].astype(float),
            width,
            label=label,
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Aggregate AP")
    ax.set_ylim(0.0, ylim)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(ncol=3, loc="upper left", frameon=True)
    ax.set_title(title)

    for idx, row in df.iterrows():
        samples = row["evaluated_samples"]
        annotation = f"{comm_labels[idx]}\nn={samples}"
        ax.text(
            idx,
            0.015,
            annotation,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    fig.tight_layout()
    fig.savefig(ROOT / f"{out_stem}.png", bbox_inches="tight")
    fig.savefig(ROOT / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_protocol() -> None:
    df = pd.read_csv(PROTOCOL_CSV)
    order = [
        "HeadOnly_attentive",
        "PureLate_attentive",
        "FullPerceptionPCS_attentive",
        "EdgeCooperHD_attentive",
        "SGCP_PAPG_attentive",
        "Full20Early_attentive",
    ]
    labels = [
        "Head-only",
        "Pure late",
        "FullPerception-PCS",
        "EdgeCooper-HD",
        "SGCP-PAPG",
        "Full 20-CAV",
    ]
    comm_labels = [
        "raw 0.0",
        "box 1.37",
        "raw 16.4",
        "raw 65.4",
        "raw 62.5",
        "raw 118.7",
    ]
    picked = df.set_index("label").loc[order].reset_index()
    grouped_bar(
        picked,
        labels,
        "Attentive Protocol-Native Aggregate AP Breakdown",
        "figure2_protocol_breakdown_attentive",
        comm_labels,
    )


def make_fusion() -> None:
    df = pd.read_csv(FUSION_CSV)
    order = [
        "HeadOnly_attentive",
        "PureLate_attentive",
        "ClusteredEarlyOnly_attentive",
        "FullSGCP_attentive",
        "OneClusterEarlyOnly_attentive",
    ]
    labels = [
        "Head-only",
        "Pure late",
        "Clustered\nearly-only",
        "Full SGCP",
        "Full 20-CAV\nearly",
    ]
    comm_labels = [
        "raw 0.0",
        "box 1.37",
        "raw 62.5",
        "raw 62.5",
        "raw 118.7",
    ]
    picked = df.set_index("label").loc[order].reset_index()
    grouped_bar(
        picked,
        labels,
        "Attentive Fusion Contribution by IoU Threshold",
        "figure3_fusion_contribution_attentive",
        comm_labels,
    )


def make_scheduler() -> None:
    df = pd.read_csv(SCHEDULER_CSV)
    order = [
        "RandomBudget_attentive",
        "DensityGreedy_attentive",
        "LinkAwareDensity_attentive",
        "PACP_LiDAR_attentive",
        "EdgeCooperHD_attentive",
        "SGCP_PAPG_attentive",
    ]
    labels = [
        "Random",
        "Density",
        "Link-aware",
        "PACP-LiDAR",
        "EdgeCooper-HD",
        "SGCP-PAPG",
    ]
    comm_labels = [
        "raw 61.2",
        "raw 75.9",
        "raw 75.9",
        "raw 86.6",
        "raw 65.4",
        "raw 62.5",
    ]
    picked = df.set_index("label").loc[order].reset_index()
    grouped_bar(
        picked,
        labels,
        "Attentive SGCP-Compatible Scheduler Comparison",
        "figure4_scheduler_comparison_attentive",
        comm_labels,
    )


def main() -> None:
    make_protocol()
    make_fusion()
    make_scheduler()


if __name__ == "__main__":
    main()
