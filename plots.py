import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path(__file__).parent / "outputs_video_poc/summary.csv"

COLORS = ["#3266ad", "#b84a2a", "#2a9d7a", "#8b5ea3", "#d4a017", "#1a7abf", "#c04a6b", "#4a8c3f"]


def load_data(path):
    df = pd.read_csv(path)
    df["crf"] = pd.to_numeric(df["crf"], errors="coerce")
    return df.dropna(subset=["crf"]).sort_values(["video", "crf"])


def make_plots(df, videos, output=None):
    panels = [
        ("compressed_size_mb", "Size (MB)"),
        ("pseudo_recall", "Recall"),
        ("pseudo_precision", "Precision"), #matched/compressed_detections
        ("mean_confidence_drop", "Mean Confidence Drop"),
        ("bpp", "Bits Per Pixel"),
        ("compression_ratio", "Size Compression"), #original_size/compressed_size
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.canvas.manager.set_window_title("Video compression analysis")
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, panels):
        for vid, color in zip(videos, COLORS):
            sub = df[df["video"] == vid]
            ax.plot(sub["crf"], sub[col], marker="o", label=vid, color=color)
        ax.set_title(title)
        ax.set_xlabel("CRF")
        ax.grid(True)
        ax.legend(fontsize=7)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print("Saved:", output)
    else:
        plt.show()


def print_summary(df, videos):
    for vid in videos:
        sub = df[df["video"] == vid]
        best = sub.loc[sub["pseudo_recall"].idxmax()]
        smallest = sub.loc[sub["compressed_size_mb"].idxmin()]
        print(f"\n{vid}")
        print(f"  best recall: CRF {int(best['crf'])} ({best['pseudo_recall']:.2f})")
        print(f"  smallest:    CRF {int(smallest['crf'])} ({smallest['compressed_size_mb']:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output", default=None)
    parser.add_argument("--video", nargs="*")
    args = parser.parse_args()

    df = load_data(args.csv)
    videos = args.video or df["video"].unique()

    print_summary(df, videos)
    make_plots(df, videos, args.output)


if __name__ == "__main__":
    main()