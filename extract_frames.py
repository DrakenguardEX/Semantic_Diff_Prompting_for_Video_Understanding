# extract_frames.py
import cv2
import os

def extract_frames(video_path: str, out_dir: str, max_frames: int = 16):
    """
    Uniformly extract frames from a single video and save them to out_dir.
    video_path: path to the input video (.webm, .mp4, etc.)
    out_dir: directory to save extracted frames
    max_frames: maximum number of frames to save
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    step = max(total // max_frames, 1)

    idx = 0
    saved = 0
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame_path = os.path.join(out_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"[extract_frames] {os.path.basename(video_path)} -> {saved} frames saved to {out_dir}")






