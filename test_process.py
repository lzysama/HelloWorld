"""Smoke test: generate a synthetic video with a shaking rectangle and run the pipeline."""
import cv2
import numpy as np
import subprocess
import sys
import os


def make_synthetic_video(path: str, frames: int = 60, w: int = 640, h: int = 480) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    rng = np.random.default_rng(42)
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Shaking "screen" rectangle with random offset
        jitter_x = int(rng.integers(-10, 10))
        jitter_y = int(rng.integers(-8, 8))
        x1, y1 = 100 + jitter_x, 80 + jitter_y
        x2, y2 = 540 + jitter_x, 400 + jitter_y
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
        # Add some content
        cv2.putText(frame, f"Frame {i:03d}", (x1 + 10, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    input_path = "/tmp/test_input.mp4"
    output_path = "/tmp/test_output.mp4"

    print("Creating synthetic test video...")
    make_synthetic_video(input_path)

    print("Running pipeline with manual ROI (no auto-detect needed)...")
    result = subprocess.run(
        [sys.executable, "process_gopro_screen.py",
         "--input", input_path,
         "--output", output_path,
         "--roi", "80,60,480,360",
         "--scale", "1",
         "--no-enhance"],
        capture_output=False
    )

    if result.returncode != 0:
        print("FAILED")
        sys.exit(1)

    assert os.path.exists(output_path) and os.path.getsize(output_path) > 1000, \
        "Output file missing or too small"

    cap = cv2.VideoCapture(output_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 60, f"Expected 60 frames, got {frame_count}"

    print(f"\nSMOKE TEST PASSED — output has {frame_count} frames at {output_path}")
