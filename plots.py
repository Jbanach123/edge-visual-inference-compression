import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

# Default directory where summary CSV files are stored
DEFAULT_DIR = Path(__file__).parent / "outputs"

# Line style and marker per resolution scale
SCALE_STYLES = {
    1.0:  ("o", "-",  "full (1.0)"),
    0.5:  ("s", "--", "half (0.5)"),
    0.25: ("^", "-.", "quarter (0.25)"),
}

# Color per resolution scale
SCALE_COLORS = {
    1.0:  "#3266ad",
    0.5:  "#2a9d7a",
    0.25: "#b84a2a",
}

# Minimum acceptable detection quality (recall / precision threshold)
QUALITY_THRESHOLD = 0.9


def find_csv(path_arg):
    """
    Resolve the CSV file to load.
    If a path is provided explicitly, use it.
    Otherwise, find the most recent summary*.csv in the default directory.
    """
    if path_arg:
        p = Path(path_arg)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p
    candidates = sorted(DEFAULT_DIR.glob("summary*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No summary*.csv found in: {DEFAULT_DIR}\n"
            f"Use --csv path/to/file.csv"
        )
    latest = candidates[-1]
    print(f"Loading: {latest}")
    return latest


def load_data(path):
    """
    Load and prepare the CSV data.

    - Parses CRF and scale as numeric values.
    - Adds a synthetic CRF=0 row for scale=1.0 only, representing the
      uncompressed original. This serves as the reference point on the X axis.
      Values: recall=1.0, precision=1.0, conf_drop=0.0, size=original_size_mb.
    """
    df = pd.read_csv(path)
    df["crf"]   = pd.to_numeric(df["crf"],   errors="coerce")
    df["scale"] = pd.to_numeric(df["scale"], errors="coerce")
    df = df.dropna(subset=["crf", "scale"]).sort_values(["video", "scale", "crf"])

    # Synthetic CRF=0 point (original video) — only for scale=1.0.
    # Half and quarter scales have no original reference so they start
    # from their lowest measured CRF value.
    originals = []
    for vid, group in df.groupby("video"):
        row = group[group["scale"] == 1.0].iloc[0]
        originals.append({
            "video":                vid,
            "scale":                1.0,
            "crf":                  0,
            "compressed_size_mb":   row["original_size_mb"],  # original = 100% size
            "pseudo_recall":        1.0,
            "pseudo_precision":     1.0,
            "mean_confidence_drop": 0.0,
        })

    df = pd.concat([pd.DataFrame(originals), df], ignore_index=True)
    return df.sort_values(["video", "scale", "crf"])


def compute_stats(df, scales):
    """
    Aggregate metrics across all videos for each (scale, CRF) combination.
    Returns mean and std for each metric, used to draw the ribbon plot.
    """
    cols = ["compressed_size_mb", "pseudo_recall", "pseudo_precision", "mean_confidence_drop"]
    df = df[df["scale"].isin(scales)]
    stats = (
        df.groupby(["scale", "crf"])[cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["scale", "crf"] + [
        f"{c}_{s}" for c in cols for s in ["mean", "std"]
    ]
    return stats


def make_plots(df, group_name, output=None):
    """
    Generate a 2×2 grid of plots:
      - Size (MB):          transmitted file size after compression/downscale
      - Recall:             fraction of reference detections preserved
      - Precision:          fraction of compressed detections matched to reference
      - Confidence drop:    mean decrease in YOLO detection confidence score

    Each panel shows mean ± std ribbon across all videos in the group.
    A horizontal threshold line marks the minimum acceptable quality (0.9).
    A vertical dashed line at CRF=0 marks the original (uncompressed) reference.
    """
    panels = [
        ("compressed_size_mb",   "Size (MB)",             None),
        ("pseudo_recall",        "Recall [0–1]",          QUALITY_THRESHOLD),
        ("pseudo_precision",     "Precision [0–1]",       QUALITY_THRESHOLD),
        ("mean_confidence_drop", "Confidence drop [0–1]", None),
    ]

    scales = sorted(
        [s for s in SCALE_STYLES if s in df["scale"].unique()], reverse=True
    )
    stats = compute_stats(df, scales)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, title, threshold) in zip(axes, panels):
        for scale in scales:
            marker, linestyle, scale_label = SCALE_STYLES[scale]
            color = SCALE_COLORS[scale]
            sub = stats[stats["scale"] == scale].sort_values("crf")
            if sub.empty:
                continue

            mean = sub[f"{col}_mean"]
            std  = sub[f"{col}_std"].fillna(0)
            crf  = sub["crf"]

            # Mean line with markers
            ax.plot(crf, mean,
                    marker=marker, linestyle=linestyle,
                    color=color, linewidth=2, markersize=6,
                    label=scale_label)

            # ± std deviation ribbon
            ax.fill_between(crf, mean - std, mean + std,
                            color=color, alpha=0.15)

        # Horizontal quality threshold line (recall / precision panels only)
        if threshold is not None:
            ax.axhline(y=threshold, color="#e63946", linestyle=":",
                       linewidth=1.4, alpha=0.9,
                       label=f"threshold ({threshold})")

        # Vertical reference line at CRF=0 (original video)
        ax.axvline(x=0, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.5)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("CRF  (0 = original)")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(left=-2)

    # ── Shared legend ─────────────────────────────────────────────────────────
    scale_handles = [
        mlines.Line2D([], [], color=SCALE_COLORS[s], marker=m, linestyle=ls,
                      linewidth=2, markersize=7, label=lbl)
        for s, (m, ls, lbl) in SCALE_STYLES.items()
        if s in scales
    ]
    band_handle = plt.Rectangle(
        (0, 0), 1, 1, fc="#888", alpha=0.2, label="± std dev"
    )
    threshold_handle = mlines.Line2D(
        [], [], color="#e63946", linestyle=":", linewidth=1.5,
        label=f"quality threshold ({QUALITY_THRESHOLD})"
    )

    fig.suptitle(
        f"Video compression analysis — {group_name} scenes"
        f"  (mean ± std, n={df['video'].nunique()})",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0.09, 1, 0.99])

    fig.legend(
        handles=scale_handles + [band_handle, threshold_handle],
        loc="lower center",
        ncol=len(scale_handles) + 2,
        fontsize=10,
        framealpha=0.95,
        edgecolor="#ccc",
        bbox_to_anchor=(0.5, 0.0),
        title="resolution scale",
        title_fontsize=9,
    )

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print("Saved:", output)
    else:
        plt.show()


def print_summary(df):
    """
    Print a terminal summary showing for each scale:
      - Sweet spot: highest CRF where mean recall still meets the threshold
      - Best recall: CRF with the highest mean recall value
    """
    df = df[df["crf"] > 0]
    print(f"\n{'scale':<10} {'sweet spot (recall≥0.9)':<28} {'best recall'}")
    print("─" * 60)
    for scale in sorted(df["scale"].unique(), reverse=True):
        sub = df[df["scale"] == scale]
        mean_recall = sub.groupby("crf")["pseudo_recall"].mean()
        above    = mean_recall[mean_recall >= QUALITY_THRESHOLD]
        sweet    = f"CRF {int(above.index.max())}" if not above.empty else "none"
        best_crf = mean_recall.idxmax()
        print(f"  {scale:<8} {sweet:<28} CRF {int(best_crf)} = {mean_recall[best_crf]:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize H.264 compression metrics vs YOLO detection quality."
    )
    parser.add_argument("--csv",    default=None,
                        help="Path to summary CSV (default: latest summary*.csv in outputs/)")
    parser.add_argument("--group",  default="crowd",
                        help="Scene group label used in the plot title (e.g. crowd, single)")
    parser.add_argument("--output", default=None,
                        help="Save plot to file (e.g. results.png). Opens window if omitted.")
    parser.add_argument("--video",  nargs="*",
                        help="Filter specific video files (e.g. --video 2.mp4 3.mp4)")
    args = parser.parse_args()

    csv_path = find_csv(args.csv)
    df       = load_data(csv_path)

    if args.video:
        df = df[df["video"].isin(args.video)]

    n = df["video"].nunique()
    print(f"\nGroup: {args.group}  |  n={n} videos")
    print_summary(df)
    make_plots(df, args.group, args.output)


if __name__ == "__main__":
    main()