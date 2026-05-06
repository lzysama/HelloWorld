"""
GoPro screen extraction engine.
Detects the screen rectangle in each frame, stabilizes it via homography tracking,
and enhances the extracted content.
"""

import cv2
import numpy as np
import os
import tempfile


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Sharpen and enhance a single frame."""
    # CLAHE for contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    frame = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)
    return frame


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_screen_contour(frame: np.ndarray):
    """
    Detect the GoPro screen rectangle using edge detection + contour analysis.
    Returns ordered 4 corner points or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Bilateral filter to smooth while keeping edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Canny edges
    edges = cv2.Canny(gray, 30, 100)

    # Dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    frame_area = h * w
    best = None
    best_area = 0

    for cnt in contours[:15]:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.05 or area > frame_area * 0.95:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            ordered = order_points(pts)
            # Aspect ratio sanity check (screen is roughly 4:3 or 16:9)
            width = np.linalg.norm(ordered[1] - ordered[0])
            height = np.linalg.norm(ordered[3] - ordered[0])
            if height == 0:
                continue
            ratio = width / height
            if 0.5 < ratio < 3.0 and area > best_area:
                best = ordered
                best_area = area

    return best


def compute_output_size(pts: np.ndarray):
    """Compute output width and height from the 4 ordered corners."""
    (tl, tr, br, bl) = pts
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    W = int(max(w1, w2))
    H = int(max(h1, h2))
    return W, H


def warp_frame(frame: np.ndarray, pts: np.ndarray, W: int, H: int) -> np.ndarray:
    """Apply perspective transform to extract the screen content."""
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (W, H))


def smooth_corners(history: list, new_pts: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average to stabilize corner positions."""
    if not history:
        return new_pts
    prev = history[-1]
    return prev * (1 - alpha) + new_pts * alpha


def refine_corners_with_flow(prev_gray, curr_gray, prev_pts: np.ndarray) -> np.ndarray | None:
    """
    Use Lucas-Kanade optical flow to track the 4 screen corners from the previous frame.
    Returns refined corner positions or None if tracking fails.
    """
    pts_flat = prev_pts.reshape(4, 1, 2).astype("float32")
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts_flat, None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if status is None or status.sum() < 4:
        return None
    return next_pts.reshape(4, 2)


def process_video(input_path: str, output_path: str, progress_callback=None) -> dict:
    """
    Main processing pipeline.
    Returns dict with info: output_path, width, height, fps, frames.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # --- Phase 1: find the screen in the first ~90 frames ---
    initial_pts = None
    scan_limit = min(90, total_frames)
    for _ in range(scan_limit):
        ret, frame = cap.read()
        if not ret:
            break
        pts = detect_screen_contour(frame)
        if pts is not None:
            initial_pts = pts
            break

    if initial_pts is None:
        cap.release()
        raise ValueError("Could not detect a screen rectangle in the video. "
                         "Make sure the GoPro screen is clearly visible.")

    out_W, out_H = compute_output_size(initial_pts)
    # Ensure even dimensions for video codec
    out_W = (out_W // 2) * 2
    out_H = (out_H // 2) * 2

    # --- Phase 2: process all frames ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_W, out_H))

    current_pts = initial_pts.copy()
    corner_history = [current_pts]
    prev_gray = None
    frame_idx = 0
    detection_fail_count = 0
    MAX_FAIL = 15  # frames before forcing re-detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Try optical flow tracking first (fast, stable)
            flow_pts = refine_corners_with_flow(prev_gray, gray, current_pts)
            if flow_pts is not None:
                detection_fail_count = 0
                smoothed = smooth_corners(corner_history, flow_pts, alpha=0.3)
                current_pts = smoothed
                corner_history.append(current_pts)
            else:
                detection_fail_count += 1

        # Fallback to full detection if flow fails
        if prev_gray is None or detection_fail_count >= MAX_FAIL:
            new_pts = detect_screen_contour(frame)
            if new_pts is not None:
                smoothed = smooth_corners(corner_history, new_pts, alpha=0.25)
                current_pts = smoothed
                corner_history.append(current_pts)
                detection_fail_count = 0

        prev_gray = gray

        # Warp + enhance
        try:
            warped = warp_frame(frame, current_pts, out_W, out_H)
            enhanced = enhance_frame(warped)
            writer.write(enhanced)
        except Exception:
            # If warp fails, write black frame to keep sync
            writer.write(np.zeros((out_H, out_W, 3), dtype=np.uint8))

        frame_idx += 1
        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx, total_frames)

    cap.release()
    writer.release()

    return {
        "output_path": output_path,
        "width": out_W,
        "height": out_H,
        "fps": fps,
        "frames": frame_idx,
    }
