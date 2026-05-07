"""
GoPro screen extractor with stabilization and quality enhancement.

Usage:
    python3 process_gopro_screen.py --input <video.mp4> [options]

Options:
    --input       Input video file (required)
    --output      Output video file (default: output_stabilized.mp4)
    --roi         Manual ROI as x,y,w,h (e.g. 100,50,640,480). Auto-detect if omitted.
    --scale       Upscale factor for quality enhancement (default: 2)
    --no-enhance  Skip image quality enhancement
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# ROI detection
# ---------------------------------------------------------------------------

def detect_gopro_screen_roi(cap: cv2.VideoCapture, sample_frames: int = 30) -> tuple[int, int, int, int]:
    """
    Auto-detect the GoPro LCD screen region from sampled frames.

    Strategy: the GoPro screen is typically a bright, high-contrast rectangle
    with sharp edges visible against the camera body / background. We look for
    the largest stable rectangle that persists across multiple frames.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_frames)

    candidate_rects: list[tuple[int, int, int, int]] = []

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Enhance edges
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        # Dilate to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_frame, w_frame = frame.shape[:2]
        min_area = (w_frame * h_frame) * 0.02   # at least 2% of frame
        max_area = (w_frame * h_frame) * 0.70   # at most 70% of frame

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            # Accept 4-sided polygons (screen-like)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / max(h, 1)
                if 0.5 < aspect < 3.0:   # reasonable screen aspect ratio
                    candidate_rects.append((x, y, w, h))

    if not candidate_rects:
        # Fallback: use entire frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        print("[WARN] Could not auto-detect screen ROI — using full frame.")
        return 0, 0, w, h

    # Cluster candidates and pick the most frequent bounding box
    rects_arr = np.array(candidate_rects, dtype=np.float32)
    # Round to 32-px grid to merge near-duplicates
    rounded = (rects_arr / 32).round() * 32
    unique, counts = np.unique(rounded, axis=0, return_counts=True)
    best = unique[np.argmax(counts)].astype(int)
    x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"[INFO] Detected ROI: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h


# ---------------------------------------------------------------------------
# Stabilization
# ---------------------------------------------------------------------------

class Stabilizer:
    """
    ECC (Enhanced Correlation Coefficient) based 2-D affine stabilizer.

    Each frame is aligned to the previous one; accumulated transforms are
    smoothed with a Gaussian window to remove high-frequency jitter while
    preserving intentional camera motion.
    """

    def __init__(self, smooth_radius: int = 30):
        self.smooth_radius = smooth_radius
        self._transforms: list[np.ndarray] = []   # per-frame 2x3 affine

    # ------------------------------------------------------------------
    # Pass 1 – collect per-frame transforms
    # ------------------------------------------------------------------

    def collect(self, frames_gray: list[np.ndarray]) -> None:
        n = len(frames_gray)
        print(f"[INFO] Collecting motion transforms for {n} frames...")
        identity = np.eye(2, 3, dtype=np.float32)
        self._transforms = [identity.copy()]

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        warp_mode = cv2.MOTION_EUCLIDEAN   # rotation + translation only

        for i in range(1, n):
            prev = frames_gray[i - 1]
            curr = frames_gray[i]
            M = identity.copy()
            try:
                _, M = cv2.findTransformECC(prev, curr, M, warp_mode, criteria)
            except cv2.error:
                M = identity.copy()
            self._transforms.append(M)

            if i % 100 == 0:
                print(f"  {i}/{n} frames analysed")

    # ------------------------------------------------------------------
    # Smooth transforms with a Gaussian kernel
    # ------------------------------------------------------------------

    def _smooth(self) -> list[np.ndarray]:
        n = len(self._transforms)
        radius = self.smooth_radius
        kernel_size = 2 * radius + 1
        kernel = cv2.getGaussianKernel(kernel_size, radius / 2.0).flatten()
        kernel /= kernel.sum()

        # Separate into 6 scalar channels and smooth each
        params = np.array([M.flatten() for M in self._transforms], dtype=np.float64)
        smoothed = np.zeros_like(params)
        for c in range(6):
            channel = params[:, c]
            conv = np.convolve(channel, kernel, mode='full')
            # 'full' output length = n + kernel_size - 1; take the centre n elements
            pad = (len(conv) - n) // 2
            smoothed[:, c] = conv[pad: pad + n]
            # Clamp boundary frames to original to avoid edge artefacts
            for e in range(min(radius, n)):
                smoothed[e, c] = channel[e]
                smoothed[n - 1 - e, c] = channel[n - 1 - e]

        return [smoothed[i].reshape(2, 3).astype(np.float32) for i in range(n)]

    # ------------------------------------------------------------------
    # Pass 2 – apply smoothed transforms
    # ------------------------------------------------------------------

    def apply(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        smoothed = self._smooth()
        h, w = frames[0].shape[:2]
        output = []
        for i, (frame, M_smooth) in enumerate(zip(frames, smoothed)):
            M_orig = self._transforms[i]
            # Correction = smooth - original (in transform space)
            # We warp with the smoothed transform and crop a safe inner region
            stabilized = cv2.warpAffine(frame, M_smooth, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
            output.append(stabilized)
        print("[INFO] Stabilization applied.")
        return output


# ---------------------------------------------------------------------------
# Quality enhancement
# ---------------------------------------------------------------------------

def enhance_frame(frame: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Upscale with Lanczos, then apply unsharp mask + bilateral denoise.
    """
    h, w = frame.shape[:2]
    up = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # Bilateral filter for noise reduction while preserving edges
    denoised = cv2.bilateralFilter(up, d=5, sigmaColor=30, sigmaSpace=30)

    # Unsharp mask for sharpening
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    return sharpened


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(input_path: str, output_path: str, roi: tuple | None,
            scale: int, do_enhance: bool) -> None:

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Input: {orig_w}x{orig_h} @ {fps:.2f} fps, {total} frames")

    # Determine ROI
    if roi:
        x, y, w, h = roi
    else:
        x, y, w, h = detect_gopro_screen_roi(cap)

    # Clamp ROI to frame bounds
    x = max(0, min(x, orig_w - 1))
    y = max(0, min(y, orig_h - 1))
    w = min(w, orig_w - x)
    h = min(h, orig_h - y)
    print(f"[INFO] Using ROI: x={x}, y={y}, w={w}, h={h}")

    # --- Pass 1: read all frames and crop to ROI ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames_color: list[np.ndarray] = []
    frames_gray: list[np.ndarray] = []

    print(f"[INFO] Reading {total} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y:y + h, x:x + w]
        frames_color.append(crop)
        frames_gray.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    cap.release()

    print(f"[INFO] Loaded {len(frames_color)} frames from ROI ({w}x{h})")

    # --- Pass 2: stabilize ---
    stab = Stabilizer(smooth_radius=30)
    stab.collect(frames_gray)
    stable_frames = stab.apply(frames_color)

    # --- Pass 3: enhance and write ---
    out_w = w * scale if do_enhance else w
    out_h = h * scale if do_enhance else h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    print(f"[INFO] Writing output {out_w}x{out_h} → {output_path}")
    for i, frame in enumerate(stable_frames):
        result = enhance_frame(frame, scale) if do_enhance else frame
        out.write(result)
        if (i + 1) % 200 == 0:
            print(f"  Written {i + 1}/{len(stable_frames)} frames")

    out.release()
    print(f"[INFO] Done. Output saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_roi(s: str) -> tuple[int, int, int, int]:
    parts = [int(v.strip()) for v in s.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return tuple(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoPro screen extractor + stabilizer + enhancer")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", default="output_stabilized.mp4", help="Output video file")
    parser.add_argument("--roi", type=parse_roi, default=None,
                        help="Manual ROI x,y,w,h (e.g. 100,50,640,480)")
    parser.add_argument("--scale", type=int, default=2,
                        help="Upscale factor for quality enhancement (default: 2)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip image quality enhancement (faster)")
    args = parser.parse_args()

    process(
        input_path=args.input,
        output_path=args.output,
        roi=args.roi,
        scale=args.scale,
        do_enhance=not args.no_enhance,
    )
