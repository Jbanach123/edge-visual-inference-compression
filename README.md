# Edge-Based Visual Inference: Video Compression vs. YOLO Detection Quality

This project studies how H.264 compression (CRF) and resolution downscaling
affect YOLO object detection quality, simulating video transmission from an
edge device to a receiver. It consists of two scripts forming a pipeline:

1. **`video_compression_analysis.py`** — runs the experiments and produces a
   results CSV.
2. **`plots.py`** — visualizes that CSV as a 4-panel chart.

## Pipeline overview

For each experiment (a `scale` × `crf` combination):

1. The source video is **downscaled** to `scale × original resolution` and
   **compressed** with H.264 at the given CRF → this is the small file that
   would actually be sent over the network. Its size and `bpp` are measured
   here.
2. The small file is **upscaled** back to the original resolution (CRF=10,
   near-lossless) → this is what the receiver sees, and YOLO runs on it.
3. Detections on the compressed/upscaled video are compared against a YOLO
   baseline run on the original video (matched by class + IoU ≥ 0.5),
   producing metrics like `pseudo_recall`, `pseudo_precision`, and
   `mean_confidence_drop`.

The default experiment matrix covers scales **1.0 / 0.5 / 0.25**, each with
9 CRF values (10–51) — 27 combinations per video.

## Requirements

```bash
pip install ultralytics opencv-python numpy pandas matplotlib
```

**ffmpeg** must also be available on the system (in PATH):

- Windows: https://ffmpeg.org/download.html
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

YOLO model weights default to `yolov8n.pt` (downloaded automatically by
`ultralytics` on first use).

## Directory structure

```
project/
    video_compression_analysis.py
    plots.py
    videos/
        clip1.mp4
        clip2.mp4
        ...
    outputs/                          # created automatically
        summary_<timestamp>.csv
        <video_name>/
            original_annotated.mp4
            <experiment>_annotated.mp4
            compressed/
                <video_name>_<experiment>.mp4
                <video_name>_<experiment>_small.mp4
```

## Usage

### 1. Run the experiments

```bash
python video_compression_analysis.py
```

No CLI arguments — configuration (input folder, model, thresholds,
experiment list) is set directly in the `SETTINGS` section at the top of
the file. Results are saved to `outputs/summary_<timestamp>.csv`.

> **Note:** the full matrix (27 combinations per video, YOLO run twice per
> combination) is time-consuming — for longer or more numerous videos this
> can take hours, mainly due to repeated frame-by-frame YOLO inference. It's
> worth testing first on a short clip or a reduced experiment list.

### 2. Visualize the results

```bash
# uses the latest summary*.csv from outputs/
python plots.py

# point to a specific CSV file
python plots.py --csv outputs/summary_20250115_143000.csv

# scene group label shown in the plot title (default "crowd")
python plots.py --group single

# save to a file instead of opening a window
python plots.py --output results.png

# only selected videos
python plots.py --video clip1.mp4 clip2.mp4
```

The plot shows 4 panels (Size, Recall, Precision, Confidence drop) as a
function of CRF, one line per resolution scale (mean ± std across videos),
with a 0.9 quality threshold marked on the Recall/Precision panels and a
vertical reference line at CRF=0 (the original, uncompressed video). The
terminal also prints a summary with the "sweet spot" CRF per scale (highest
CRF where recall still meets the threshold) and the CRF giving the best
recall.

## CLI arguments (`plots.py`)

| Argument | Default | Description |
|---|---|---|
| `--csv` | latest `summary*.csv` in `outputs/` | path to the CSV file |
| `--group` | `crowd` | scene group label shown in the plot title |
| `--output` | none (opens a window) | path to save the plot, e.g. `results.png` |
| `--video` | none (all) | filter specific video files by name |

## Output CSV columns

| Column | Meaning |
|---|---|
| `scale`, `crf` | experiment parameters |
| `tx_resolution`, `pixel_reduction` | transmitted resolution and pixel-count reduction |
| `bpp` | bits per pixel (transmission cost, measured on the small file) |
| `compressed_size_mb`, `compression_ratio` | size of the transmitted file and compression ratio |
| `pseudo_recall` / `pseudo_precision` | fraction of preserved / valid detections vs. baseline |
| `mean_confidence_drop` | average drop in YOLO detection confidence |
| `missed_detections` / `new_detections` | detections lost / falsely added |
