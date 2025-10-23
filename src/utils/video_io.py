import cv2, os, math, glob
from pathlib import Path

def list_video_files(input_dir):
    exts = ('*.mp4','*.mov','*.avi','*.mkv','*.MP4')
    files = []
    for e in exts:
        files.extend(Path(input_dir).glob(e))
    return sorted(files)

def open_video_reader(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h

def make_video_writer(out_path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))

def evenly_spaced_indices(n, k=6):
    if n <= k:
        return list(range(n))
    step = n / float(k+1)
    return [int((i+1)*step) for i in range(k)]
