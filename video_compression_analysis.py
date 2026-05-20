"""
Edge-based Visual Inference — impact of H.264 video compression on YOLO detection.

Research metrics:
    - bpp (bits per pixel)          — transmission cost per pixel
    - pseudo_recall                 — proxy for detection recall (matched / reference)
    - missed_detections             — number of objects lost after compression
    - mean_confidence_drop          — average drop in YOLO confidence score

Pipeline per experiment:
    1. Downscale the video to the target resolution (scale factor)
    2. Compress with H.264 at the given CRF value  →  small file (transmitted)
    3. Upscale back to the original resolution      →  YOLO receives full-size frame
    bpp and compressed_size_mb are measured from the small (transmitted) file.
    Detection metrics are measured on the upscaled file (what the receiver sees).

Requirements:
    pip install ultralytics opencv-python numpy

    ffmpeg must be available on the system:
        Windows:  https://ffmpeg.org/download.html  (add to PATH)
        Linux:    sudo apt install ffmpeg
        macOS:    brew install ffmpeg

Directory structure:
    project/
        video_compression_analysis.py
        videos/
            clip1.mp4
            clip2.mp4
            ...

Usage:
    python video_compression_analysis.py
"""

import csv
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_DIR  = Path("videos")        # folder with source video files
OUTPUT_DIR = Path("outputs")  # root output folder

MODEL_PATH = "yolov8n.pt"          # YOLOv8 model weights

YOLO_CONF     = 0.25   # minimum confidence threshold for YOLO detections
IOU_THRESHOLD = 0.5    # minimum IoU to consider two bounding boxes as matching

# Experiment matrix: each entry defines one (scale, CRF) combination.
# scale = resolution multiplier (0.5 → half width & height → 4× fewer pixels)
# crf   = H.264 Constant Rate Factor (higher = more compression, lower quality)
EXPERIMENTS = [
    # ── full resolution ──────────────────────────────────────────────────────
    {"scale": 1.0,   "crf": 10, "name": "full_crf10"},
    {"scale": 1.0,   "crf": 15, "name": "full_crf15"},
    {"scale": 1.0,   "crf": 20, "name": "full_crf20"},
    {"scale": 1.0,   "crf": 25, "name": "full_crf25"},
    {"scale": 1.0,   "crf": 30, "name": "full_crf30"},
    {"scale": 1.0,   "crf": 35, "name": "full_crf35"},
    {"scale": 1.0,   "crf": 40, "name": "full_crf40"},
    {"scale": 1.0,   "crf": 45, "name": "full_crf45"},
    {"scale": 1.0,   "crf": 51, "name": "full_crf51"},
    # ── half resolution (4× fewer pixels) ───────────────────────────────────
    {"scale": 0.5,   "crf": 10, "name": "half_crf10"},
    {"scale": 0.5,   "crf": 15, "name": "half_crf15"},
    {"scale": 0.5,   "crf": 20, "name": "half_crf20"},
    {"scale": 0.5,   "crf": 25, "name": "half_crf25"},
    {"scale": 0.5,   "crf": 30, "name": "half_crf30"},
    {"scale": 0.5,   "crf": 35, "name": "half_crf35"},
    {"scale": 0.5,   "crf": 40, "name": "half_crf40"},
    {"scale": 0.5,   "crf": 45, "name": "half_crf45"},
    {"scale": 0.5,   "crf": 51, "name": "half_crf51"},
    # ── quarter resolution (16× fewer pixels) ────────────────────────────────
    {"scale": 0.25,  "crf": 10, "name": "quarter_crf10"},
    {"scale": 0.25,  "crf": 15, "name": "quarter_crf15"},
    {"scale": 0.25,  "crf": 20, "name": "quarter_crf20"},
    {"scale": 0.25,  "crf": 25, "name": "quarter_crf25"},
    {"scale": 0.25,  "crf": 30, "name": "quarter_crf30"},
    {"scale": 0.25,  "crf": 35, "name": "quarter_crf35"},
    {"scale": 0.25,  "crf": 40, "name": "quarter_crf40"},
    {"scale": 0.25,  "crf": 45, "name": "quarter_crf45"},
    {"scale": 0.25,  "crf": 51, "name": "quarter_crf51"},
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_video_info(video_path):
    """Return (width, height, fps, frame_count) for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, frames


def compress_and_resize(input_path, output_path, scale, crf):
    """
    Simulate edge-device transmission with resolution reduction:

      Step 1 — Downscale to (scale × original resolution) and compress with
               H.264 at the given CRF. This produces the *small file* that
               would be sent over the network. Size and bpp are measured here.

      Step 2 — Upscale the small file back to the original resolution using
               CRF=10 (near-lossless). YOLO runs on this upscaled file, which
               represents the degraded image the receiver processes.

    For scale=1.0 only CRF compression is applied; no resize is needed.

    Returns:
        Path to the small (transmitted) file — used for size and bpp measurement.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orig_w, orig_h, _, _ = get_video_info(input_path)

    if scale == 1.0:
        # No resolution change — compress only
        command = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-an",                  # strip audio
            str(output_path),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"FFmpeg error: {input_path} scale={scale} crf={crf}")
        # For scale=1.0 the output file IS the transmitted file
        return output_path

    else:
        # Ensure dimensions are divisible by 2 (H.264 requirement)
        small_w = max(2, int(orig_w * scale) // 2 * 2)
        small_h = max(2, int(orig_h * scale) // 2 * 2)

        # Step 1: downscale + compress → small file (what is transmitted)
        small_path = output_path.with_name(output_path.stem + "_small.mp4")
        command_down = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"scale={small_w}:{small_h}",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-an",
            str(small_path),
        ]
        result = subprocess.run(command_down, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"FFmpeg error (down): {input_path} scale={scale} crf={crf}")

        # Step 2: upscale back to original resolution → YOLO input
        # CRF=10 here to avoid adding extra compression artefacts on top
        command_up = [
            "ffmpeg", "-y",
            "-i", str(small_path),
            "-vf", f"scale={orig_w}:{orig_h}",
            "-c:v", "libx264",
            "-crf", "10",
            "-an",
            str(output_path),
        ]
        result = subprocess.run(command_up, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"FFmpeg error (up): {input_path} scale={scale} crf={crf}")

        # Return the small file — size and bpp must be measured from it
        return small_path


def compute_bpp(video_path):
    """
    Compute average bits per pixel (bpp) for a video file.

        bpp = file_size_in_bits / (width × height × frame_count)

    Always called on the small (transmitted) file so that bpp reflects
    the actual transmission cost, not the upscaled output.
    Lower bpp = higher compression = less data sent over the network.
    """
    cap = cv2.VideoCapture(str(video_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    total_pixels = width * height * max(frames, 1)
    total_bits   = video_path.stat().st_size * 8
    return total_bits / total_pixels


def box_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) for two bounding boxes [x1, y1, x2, y2].
    Returns a value in [0, 1]; higher = greater overlap.
    Used to decide whether two detections refer to the same object.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def run_yolo_on_video(model, video_path, output_annotated_video=None):
    """
    Run YOLO inference on every frame of a video.

    Returns:
        dict {frame_id: [list of detections]}
        Each detection: {"cls": int, "conf": float, "box": [x1, y1, x2, y2]}

    Optionally saves an annotated video with bounding boxes drawn.
    Called twice per experiment:
        1. On the original video  → reference detections (baseline)
        2. On the upscaled file   → compressed detections (to compare)
    """
    cap = cv2.VideoCapture(str(video_path))
    detections_by_frame = {}
    frame_id = 0
    writer = None
    temp_path = None

    if output_annotated_video is not None:
        output_annotated_video.parent.mkdir(parents=True, exist_ok=True)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_path = output_annotated_video.with_name(
            output_annotated_video.stem + "_temp.mp4"
        )
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height)
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=YOLO_CONF, verbose=False)[0]
        dets = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                dets.append({
                    "cls":  int(cls),
                    "conf": float(conf),
                    "box":  box.tolist(),
                })

        detections_by_frame[frame_id] = dets

        if writer is not None:
            writer.write(result.plot())

        frame_id += 1
        if frame_id % 100 == 0:
            print(f"      frame {frame_id}...")

    cap.release()

    if writer is not None:
        writer.release()
        # Re-encode with ffmpeg for proper MP4 compatibility
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(temp_path),
            "-c:v", "libx264", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an",
            str(output_annotated_video),
        ], check=True)
        temp_path.unlink()

    print(f"      total: {frame_id} frames")
    return detections_by_frame


def match_detections(reference_dets, compressed_dets):
    """
    Greedy matching of detections between the reference and compressed frame.

    Rules:
    - Only boxes of the same class are compared.
    - IoU must be >= IOU_THRESHOLD to be considered a valid match.
    - Each box can be matched at most once (greedy by highest IoU first).

    Returns:
        List of matched pairs: [{"iou", "ref", "cmp"}, ...]
    """
    candidates = []
    for ri, rd in enumerate(reference_dets):
        for ci, cd in enumerate(compressed_dets):
            if rd["cls"] != cd["cls"]:
                continue
            iou = box_iou(rd["box"], cd["box"])
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, ri, ci))

    candidates.sort(reverse=True)   # process highest IoU first
    used_ref, used_cmp, matches = set(), set(), []

    for iou, ri, ci in candidates:
        if ri in used_ref or ci in used_cmp:
            continue
        used_ref.add(ri); used_cmp.add(ci)
        matches.append({
            "iou": iou,
            "ref": reference_dets[ri],
            "cmp": compressed_dets[ci],
        })

    return matches


def compare_videos(reference_results, compressed_results):
    """
    Frame-by-frame comparison of detections between original and compressed video.

    Metrics computed:
        pseudo_recall    = matched / reference_detections
                           (fraction of original detections preserved)
        pseudo_precision = matched / compressed_detections
                           (fraction of compressed detections that are valid)
        missed_detections  = reference boxes with no match in compressed
        new_detections     = compressed boxes with no match in reference (false positives)
        mean_confidence_drop = mean(conf_reference - conf_compressed) for matched pairs
    """
    total_ref = total_cmp = total_matches = total_missed = total_new = 0
    conf_drops        = []
    conf_baseline_l   = []
    conf_compressed_l = []

    frame_ids = sorted(set(reference_results) & set(compressed_results))

    for fid in frame_ids:
        ref_dets = reference_results[fid]
        cmp_dets = compressed_results[fid]
        matches  = match_detections(ref_dets, cmp_dets)

        total_ref     += len(ref_dets)
        total_cmp     += len(cmp_dets)
        total_matches += len(matches)
        total_missed  += len(ref_dets) - len(matches)
        total_new     += len(cmp_dets) - len(matches)

        for m in matches:
            rc = m["ref"]["conf"]
            cc = m["cmp"]["conf"]
            conf_drops.append(rc - cc)
            conf_baseline_l.append(rc)
            conf_compressed_l.append(cc)

    pseudo_recall    = total_matches / total_ref if total_ref > 0 else 0.0
    pseudo_precision = total_matches / total_cmp if total_cmp > 0 else 0.0

    avg_conf_baseline   = float(np.mean(conf_baseline_l))   if conf_baseline_l   else 0.0
    avg_conf_compressed = float(np.mean(conf_compressed_l)) if conf_compressed_l else 0.0
    mean_conf_drop      = float(np.mean(conf_drops))        if conf_drops        else 0.0

    return {
        "frames_compared":       len(frame_ids),
        "reference_detections":  total_ref,
        "compressed_detections": total_cmp,
        "matched_detections":    total_matches,
        "missed_detections":     total_missed,
        "avg_conf_baseline":     round(avg_conf_baseline,   4),
        "avg_conf_compressed":   round(avg_conf_compressed, 4),
        "mean_confidence_drop":  round(mean_conf_drop,      4),
        "pseudo_recall":         round(pseudo_recall,       4),
        "new_detections":        total_new,
        "pseudo_precision":      round(pseudo_precision,    4),
    }


def file_size_mb(path):
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)

    video_paths = sorted(INPUT_DIR.glob("*.mp4"))
    if not video_paths:
        print(f"No .mp4 files found in: {INPUT_DIR}")
        return

    print(f"Found {len(video_paths)} videos, "
          f"{len(EXPERIMENTS)} experiments = "
          f"{len(video_paths) * len(EXPERIMENTS)} combinations\n")

    summary_rows = []

    for video_path in video_paths:
        print("=" * 70)
        print(f"VIDEO: {video_path.name}")

        video_name     = video_path.stem
        vid_out_dir    = OUTPUT_DIR / video_name
        compressed_dir = vid_out_dir / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)

        orig_w, orig_h, _, _ = get_video_info(video_path)
        original_size_mb = file_size_mb(video_path)
        original_bpp     = compute_bpp(video_path)

        print(f"  Resolution: {orig_w}x{orig_h} | "
              f"Size: {original_size_mb:.1f} MB | bpp: {original_bpp:.4f}")
        print("  Running YOLO on original (baseline)...")

        reference_results = run_yolo_on_video(
            model,
            video_path,
            output_annotated_video=vid_out_dir / "original_annotated.mp4"
        )

        for exp in EXPERIMENTS:
            name  = exp["name"]
            scale = exp["scale"]
            crf   = exp["crf"]

            tx_w = max(2, int(orig_w * scale) // 2 * 2)
            tx_h = max(2, int(orig_h * scale) // 2 * 2)
            pixel_reduction = round(1 / (scale ** 2))

            compressed_path = compressed_dir / f"{video_name}_{name}.mp4"

            print(f"\n  [{name}] "
                  f"tx {tx_w}x{tx_h} "
                  f"(x{pixel_reduction} fewer pixels) | CRF={crf}")

            # small_path = the file actually transmitted over the network
            # compressed_path = upscaled version used for YOLO inference
            small_path = compress_and_resize(video_path, compressed_path, scale, crf)

            # Measure size and bpp from the transmitted (small) file
            comp_size_mb = file_size_mb(small_path)
            bpp          = compute_bpp(small_path)
            ratio        = original_size_mb / comp_size_mb if comp_size_mb > 0 else 0

            print(f"    Transmitted size: {comp_size_mb:.2f} MB | "
                  f"ratio: {ratio:.1f}x | bpp: {bpp:.4f}")
            print(f"    Running YOLO on compressed...")

            compressed_results = run_yolo_on_video(
                model,
                compressed_path,
                output_annotated_video=vid_out_dir / f"{name}_annotated.mp4"
            )

            metrics = compare_videos(reference_results, compressed_results)

            row = {
                "video":              video_path.name,
                "experiment":         name,
                "scale":              scale,
                "crf":                crf,
                "tx_resolution":      f"{tx_w}x{tx_h}",
                "pixel_reduction":    pixel_reduction,
                "bpp":                round(bpp, 4),
                "original_size_mb":   round(original_size_mb, 2),
                "compressed_size_mb": round(comp_size_mb, 2),   # size of transmitted file
                "compression_ratio":  round(ratio, 2),
                **metrics,
            }

            summary_rows.append(row)

            print(
                f"    missed={metrics['missed_detections']} | "
                f"recall={metrics['pseudo_recall']:.3f} | "
                f"conf: {metrics['avg_conf_baseline']:.3f}"
                f" -> {metrics['avg_conf_compressed']:.3f} "
                f"(drop={metrics['mean_confidence_drop']:+.3f})"
            )

    # Save results with a timestamp to avoid overwriting previous runs
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUTPUT_DIR / f"summary_{timestamp}.csv"

    fieldnames = [
        "video", "experiment", "scale", "crf",
        "tx_resolution", "pixel_reduction",
        "bpp", "original_size_mb", "compressed_size_mb", "compression_ratio",
        "frames_compared",
        "reference_detections", "compressed_detections",
        "matched_detections",
        "missed_detections",
        "avg_conf_baseline", "avg_conf_compressed", "mean_confidence_drop",
        "pseudo_recall",
        "new_detections", "pseudo_precision",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print(f"Done! Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
