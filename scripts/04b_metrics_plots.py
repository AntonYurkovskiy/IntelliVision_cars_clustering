"""Generate metrics overview plots from clustering_summary.csv.

These figures are useful for presentations and for the analysis notebook.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import FIGURES_DIR, RESULTS_DIR


def rank_configurations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ch_norm"] = (df["calinski_harabasz"] - df["calinski_harabasz"].min()) / (
        df["calinski_harabasz"].max() - df["calinski_harabasz"].min() + 1e-9
    )
    df["db_norm"] = (df["davies_bouldin"].max() - df["davies_bouldin"]) / (
        df["davies_bouldin"].max() - df["davies_bouldin"].min() + 1e-9
    )
    df["score"] = df["ch_norm"] + df["db_norm"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def plot_metrics_by_descriptor(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for desc, grp in summary.groupby("descriptor"):
        grp_sorted = grp.sort_values("n_clusters")
        ax.plot(grp_sorted["n_clusters"], grp_sorted["calinski_harabasz"], marker="o", markersize=4, label=desc)
    ax.set_title("Calinski-Harabasz (выше = лучше)")
    ax.set_xlabel("Число кластеров")
    ax.set_ylabel("CH Index")
    ax.legend(fontsize=9)

    ax = axes[1]
    for desc, grp in summary.groupby("descriptor"):
        grp_sorted = grp.sort_values("n_clusters")
        ax.plot(grp_sorted["n_clusters"], grp_sorted["davies_bouldin"], marker="o", markersize=4, label=desc)
    ax.set_title("Davies-Bouldin (ниже = лучше)")
    ax.set_xlabel("Число кластеров")
    ax.set_ylabel("DB Index")
    ax.legend(fontsize=9)

    fig.suptitle("Метрики кластеризации по дескрипторам и числу кластеров", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_by_descriptor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_heatmap(summary: pd.DataFrame):
    pivot_ch = (
        summary.groupby(["descriptor", "n_clusters"])["calinski_harabasz"]
        .max()
        .unstack("n_clusters")
    )
    pivot_db = (
        summary.groupby(["descriptor", "n_clusters"])["davies_bouldin"]
        .min()
        .unstack("n_clusters")
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    sns.heatmap(
        pivot_ch, ax=axes[0], cmap="YlGn", fmt=".0f", annot=True,
        linewidths=0.5, cbar_kws={"label": "CH (max)"},
    )
    axes[0].set_title("CH Index — лучшее по скейлеру и методу")
    axes[0].set_ylabel("Дескриптор")

    sns.heatmap(
        pivot_db, ax=axes[1], cmap="YlOrRd_r", fmt=".2f", annot=True,
        linewidths=0.5, cbar_kws={"label": "DB (min)"},
    )
    axes[1].set_title("DB Index — лучшее по скейлеру и методу")
    axes[1].set_ylabel("")

    fig.suptitle("Тепловые карты метрик кластеризации", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_method_boxplot(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    summary.boxplot(column="calinski_harabasz", by="method", ax=axes[0])
    axes[0].set_title("CH Index по методу")
    axes[0].set_xlabel("Метод")
    axes[0].set_ylabel("CH Index")

    summary.boxplot(column="davies_bouldin", by="method", ax=axes[1])
    axes[1].set_title("DB Index по методу")
    axes[1].set_xlabel("Метод")
    axes[1].set_ylabel("DB Index")

    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_by_method_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ch_vs_db_scatter(summary: pd.DataFrame):
    ranked = rank_configurations(summary)
    N_TOP = 4

    fig, ax = plt.subplots(figsize=(10, 7))
    descriptors = summary["descriptor"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(descriptors)))

    for desc, color in zip(descriptors, colors):
        sub = summary[summary["descriptor"] == desc]
        ax.scatter(
            sub["calinski_harabasz"], sub["davies_bouldin"],
            label=desc, alpha=0.7, s=50, color=color,
        )

    for _, row in ranked.head(N_TOP).iterrows():
        ax.annotate(
            f"{row['descriptor']}\n{row['scaler']}/{row['method']}/k={int(row['n_clusters'])}",
            xy=(row["calinski_harabasz"], row["davies_bouldin"]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    ax.set_xlabel("Calinski-Harabasz (выше = лучше)")
    ax.set_ylabel("Davies-Bouldin (ниже = лучше)")
    ax.set_title("CH vs DB для всех конфигураций кластеризации")
    ax.legend(fontsize=9, title="Дескриптор")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ch_vs_db_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS_DIR / "clustering_summary_max_components.csv")

    plot_metrics_by_descriptor(summary)
    plot_metrics_heatmap(summary)
    plot_metrics_by_method_boxplot(summary)
    plot_ch_vs_db_scatter(summary)

    print(f"Saved metrics plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
