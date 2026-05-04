"""
Edge-based Visual Inference — wpływ kompresji wideo H.264 na detekcję YOLO.

Metryki badawcze:
    - bpp (bits per pixel)
    - spadek mAP (pseudo_recall jako proxy)
    - liczba zagubionych obiektów (missed_detections)
    - zmiana Confidence Score (avg_conf_baseline vs avg_conf_compressed, delta)

Wymagania:
    pip install ultralytics opencv-python numpy

    ffmpeg musi być dostępny w systemie:
        Windows:  https://ffmpeg.org/download.html  (dodaj do PATH)
        Linux:    sudo apt install ffmpeg
        macOS:    brew install ffmpeg

Struktura katalogów:
    project/
        video_poc.py
        videos/
            film1.mp4
            film2.mp4
            ...

Uruchomienie:
    python video_poc.py
"""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# USTAWIENIA
# =============================================================================

INPUT_DIR  = Path("videos")
OUTPUT_DIR = Path("outputs_video_poc")

MODEL_PATH = "yolov8n.pt"

YOLO_CONF     = 0.25   # minimalny próg confidence dla YOLO
IOU_THRESHOLD = 0.5    # próg IoU do matchowania detekcji

# Szeroki zakres CRF — od prawie bezstratnego (10) do maksymalnej kompresji (51)
# Pozwoli zobaczyć "cliff point", czyli moment gdzie detekcja zaczyna się sypać
COMPRESSION_LEVELS = [
    {"name": "crf10", "crf": 10},   # prawie bez strat — referencyjna jakość
    {"name": "crf18", "crf": 18},   # wysoka jakość
    {"name": "crf23", "crf": 23},   # domyślny H.264
    {"name": "crf28", "crf": 28},   # średnia jakość
    {"name": "crf35", "crf": 35},   # niska jakość
    {"name": "crf42", "crf": 42},   # bardzo niska jakość
    {"name": "crf51", "crf": 51},   # maksymalna kompresja H.264
]


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def compress_video(input_path, output_path, crf):
    """
    Kompresuje całe wideo za pomocą H.264 (libx264).

    Parametr CRF (Constant Rate Factor):
        0  = bezstratna (ogromny plik)
        18 = wysoka jakość
        23 = domyślna wartość ffmpeg
        51 = najniższa jakość (najmniejszy plik)

    Zachowuje oryginalną rozdzielczość i FPS.
    Usuwa ścieżkę audio (-an) bo nas nie interesuje.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-an",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"FFmpeg error: {input_path}")


def compute_bpp(video_path):
    """
    Oblicza średnie bpp (bits per pixel) dla całego wideo.

    bpp = (rozmiar pliku w bitach) / (szerokość × wysokość × liczba klatek)

    Niska wartość bpp = duża kompresja = mało danych przesłanych przez łącze.
    Wysoka wartość bpp = mała kompresja = dużo danych.
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
    Intersection over Union dla dwóch bounding boxów [x1, y1, x2, y2].
    Wartość 0–1, im wyższe tym większe nakładanie się.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def run_yolo_on_video(model, video_path, output_annotated_video=None):
    """
    Uruchamia YOLO na każdej klatce wideo.

    Zwraca słownik: {frame_id: [lista detekcji]}
    Każda detekcja: {"cls": int, "conf": float, "box": [x1,y1,x2,y2]}

    Opcjonalnie zapisuje wideo z narysowanymi bboxami.
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
            print(f"      klatka {frame_id}...")

    cap.release()

    if writer is not None:
        writer.release()
        # Konwertuj temp plik do poprawnego MP4 przez ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(temp_path),
            "-c:v", "libx264", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an",
            str(output_annotated_video),
        ], check=True)
        temp_path.unlink()

    print(f"      łącznie: {frame_id} klatek")
    return detections_by_frame


def match_detections(reference_dets, compressed_dets):
    """
    Greedy matching detekcji: dopasowuje bbox z oryginału do bbox ze skompresowanego.

    Zasady:
    - tylko ta sama klasa obiektu,
    - IoU >= IOU_THRESHOLD,
    - każdy bbox może być dopasowany tylko raz,
    - priorytet: najwyższe IoU.

    Zwraca listę dopasowań: [{"iou", "ref", "cmp"}, ...]
    """
    candidates = []
    for ri, rd in enumerate(reference_dets):
        for ci, cd in enumerate(compressed_dets):
            if rd["cls"] != cd["cls"]:
                continue
            iou = box_iou(rd["box"], cd["box"])
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, ri, ci))

    candidates.sort(reverse=True)
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
    Klatka po klatce porównuje detekcje oryginału i wersji skompresowanej.

    Zwraca słownik ze wszystkimi metrykami badawczymi:

    TWOJE METRYKI:
        bpp                  — koszt transmisji (dodawany w main)
        missed_detections    — liczba zagubionych obiektów
        mean_confidence_drop — średni spadek confidence
        pseudo_recall        — proxy dla mAP (ile detekcji bazowych przeżyło)

    DODATKOWE:
        avg_conf_baseline    — średnie confidence na oryginale
        avg_conf_compressed  — średnie confidence po kompresji
        pseudo_precision     — ile detekcji z kompresji ma odpowiednik w oryginale
    """
    total_ref = total_cmp = total_matches = total_missed = total_new = 0

    conf_drops       = []
    conf_baseline_l  = []
    conf_compressed_l = []

    frame_ids = sorted(set(reference_results) & set(compressed_results))

    for fid in frame_ids:
        ref_dets = reference_results[fid]
        cmp_dets = compressed_results[fid]
        matches  = match_detections(ref_dets, cmp_dets)

        total_ref     += len(ref_dets)
        total_cmp     += len(cmp_dets)
        total_matches += len(matches)
        total_missed  += len(ref_dets) - len(matches)   # zagubione obiekty
        total_new     += len(cmp_dets) - len(matches)   # "nowe" (false positives)

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
        # ── TWOJE METRYKI ────────────────────────────────────────
        "missed_detections":     total_missed,        # zagubione obiekty
        "avg_conf_baseline":     round(avg_conf_baseline,   4),
        "avg_conf_compressed":   round(avg_conf_compressed, 4),
        "mean_confidence_drop":  round(mean_conf_drop,      4),  # delta conf
        "pseudo_recall":         round(pseudo_recall,        4),  # proxy mAP
        # ── DODATKOWE ────────────────────────────────────────────
        "new_detections":        total_new,
        "pseudo_precision":      round(pseudo_precision, 4),
    }


def file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


# =============================================================================
# GŁÓWNY PROGRAM
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)

    video_paths = sorted(INPUT_DIR.glob("*.mp4"))
    if not video_paths:
        print(f"Brak plików .mp4 w katalogu: {INPUT_DIR}")
        return

    summary_rows = []

    for video_path in video_paths:
        print("=" * 70)
        print(f"WIDEO: {video_path.name}")

        video_name   = video_path.stem
        vid_out_dir  = OUTPUT_DIR / video_name
        compressed_dir = vid_out_dir / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)

        original_size_mb  = file_size_mb(video_path)
        original_bpp      = compute_bpp(video_path)

        print(f"  Rozmiar oryginału: {original_size_mb:.1f} MB | bpp: {original_bpp:.4f}")
        print("  YOLO na oryginale (baseline)...")

        reference_results = run_yolo_on_video(
            model, video_path,
            output_annotated_video=vid_out_dir / "original_annotated.mp4"
        )

        for setting in COMPRESSION_LEVELS:
            cname = setting["name"]
            crf   = setting["crf"]

            compressed_path = compressed_dir / f"{video_name}_{cname}.mp4"

            print(f"\n  [{cname}] Kompresja CRF={crf}...")
            compress_video(video_path, compressed_path, crf)

            compressed_size_mb = file_size_mb(compressed_path)
            compression_ratio  = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0
            bpp                = compute_bpp(compressed_path)

            print(f"    Rozmiar: {compressed_size_mb:.1f} MB | "
                  f"ratio: {compression_ratio:.1f}x | bpp: {bpp:.4f}")
            print(f"    YOLO na skompresowanym wideo...")

            compressed_results = run_yolo_on_video(
                model, compressed_path,
                output_annotated_video=vid_out_dir / f"{cname}_annotated.mp4"
            )

            metrics = compare_videos(reference_results, compressed_results)

            row = {
                "video":              video_path.name,
                "compression":        cname,
                "crf":                crf,
                # ── koszt transmisji ─────────────────────────────
                "bpp":                round(bpp, 4),
                "original_size_mb":   round(original_size_mb,  2),
                "compressed_size_mb": round(compressed_size_mb, 2),
                "compression_ratio":  round(compression_ratio,  2),
                # ── metryki detekcji ──────────────────────────────
                **metrics,
            }

            summary_rows.append(row)

            print(
                f"    missed={metrics['missed_detections']} | "
                f"pseudo_recall={metrics['pseudo_recall']:.3f} | "
                f"conf_drop={metrics['mean_confidence_drop']:+.3f} | "
                f"avg_conf: {metrics['avg_conf_baseline']:.3f} → {metrics['avg_conf_compressed']:.3f}"
            )

    # ── Zapis CSV ─────────────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "summary.csv"
    fieldnames = [
        "video", "compression", "crf",
        # koszt transmisji
        "bpp", "original_size_mb", "compressed_size_mb", "compression_ratio",
        # techniczne
        "frames_compared", "reference_detections", "compressed_detections",
        "matched_detections",
        # TWOJE METRYKI
        "missed_detections",
        "avg_conf_baseline", "avg_conf_compressed", "mean_confidence_drop",
        "pseudo_recall",
        # dodatkowe
        "new_detections", "pseudo_precision",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print(f"Gotowe! Wyniki zapisane w: {summary_path}")
    print(f"Pliki wyjściowe:           {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
