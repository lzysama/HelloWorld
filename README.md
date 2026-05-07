# HelloWorld

## Chapter 1
*text*

~another text~
## Chapter 2
## Chapter 3
## Chapter 4

---

## GoPro Screen Extractor

Extracts the GoPro LCD screen region from a video, stabilises the shake, and
enhances image quality.

### Install

```bash
pip install -r requirements.txt
# also requires ffmpeg on PATH
```

### Run

```bash
# Auto-detect screen region, upscale 2×, stabilise
python3 process_gopro_screen.py --input your_video.mp4

# Specify ROI manually (faster, more reliable)
python3 process_gopro_screen.py --input your_video.mp4 --roi 120,60,640,480

# Skip upscale (faster)
python3 process_gopro_screen.py --input your_video.mp4 --no-enhance --scale 1
```

Output is saved to `output_stabilized.mp4` by default.

### How it works

1. **ROI detection** – samples frames, finds rectangular contours, picks the most
   persistent candidate as the screen bounding box.
2. **Stabilisation** – ECC-based euclidean motion estimation per frame; transforms
   are smoothed with a Gaussian kernel (radius 30 frames) to suppress jitter
   while preserving deliberate movement.
3. **Quality enhancement** – Lanczos upscale → bilateral denoising → unsharp mask.
