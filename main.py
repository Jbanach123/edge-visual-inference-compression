import cv2
import csv
from ultralytics import YOLO
import os
import time


def compress_image(img, quality):
    # Set JPEG compression quality (0–100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # Encode image to JPEG format in memory
    result, encimg = cv2.imencode('.jpg', img, encode_param)

    # Decode back to image (simulates compression-decompression)
    return cv2.imdecode(encimg, 1)


# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load YOLO model (pretrained)
model = YOLO("yolov8n.pt")

# Load input image
img = cv2.imread("images/test.jpg")

# Different compression levels to test
qualities = [100, 80, 60, 40, 20, 10, 5, 0]

# Confidence threshold
threshold = 0.5

# Open CSV file for writing
with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow([
        "quality",
        "size_kb",
        "detections_total",
        "detections_filtered",
        "avg_conf",
        "inference_time"
    ])

    # Experiment loop
    for q in qualities:
        compressed = compress_image(img, q)

        # Measure inference time
        start = time.time()
        results = model(compressed)
        end = time.time()

        inference_time = end - start

        # Annotated image
        annotated = results[0].plot()

        # Save image
        cv2.imwrite(f"outputs/det_q{q}.jpg", annotated)

        # Count detections
        boxes = results[0].boxes
        detections = len(boxes)

        # Confidence scores
        confidences = boxes.conf

        # Average confidence
        avg_conf = float(confidences.mean()) if len(confidences) > 0 else 0

        # Filtered detections (confidence > threshold)
        valid = confidences > threshold
        filtered_detections = int(valid.sum()) if len(confidences) > 0 else 0

        # Compute size
        _, enc = cv2.imencode('.jpg', compressed)
        size_kb = len(enc) / 1024

        # Save to CSV
        writer.writerow([
            q,
            size_kb,
            detections,
            filtered_detections,
            avg_conf,
            inference_time
        ])

        print(
            f"Q={q} | det={detections} | filtered={filtered_detections} | "
            f"conf={avg_conf:.2f} | size={size_kb:.1f}KB | time={inference_time:.3f}s"
        )