import argparse, os, time, json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from .detect import YoloDetector
from .keypoints import YoloPoseEstimator
from .track import SimpleTracker
from .utils.video_io import list_video_files, open_video_reader, make_video_writer, evenly_spaced_indices
from .utils.visualize import draw_box, draw_skeleton

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', type=str, required=True, help='Folder with local videos')
    ap.add_argument('--out_dir', type=str, default='outputs')
    ap.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda:0')
    ap.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    ap.add_argument('--save_screens', action='store_true', help='force saving screenshots (default: auto)')
    return ap.parse_args()

def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    (out_dir / 'annotated').mkdir(parents=True, exist_ok=True)
    (out_dir / 'screenshots').mkdir(parents=True, exist_ok=True)
    (out_dir / 'csv').mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics').mkdir(parents=True, exist_ok=True)

    videos = list_video_files(in_dir)
    if not videos:
        print(f'No videos found in {in_dir}. Put files like .mp4 there.')
        return

    detector = YoloDetector(device=args.device, conf=args.conf)
    poser    = YoloPoseEstimator(device=args.device, conf=args.conf)
    tracker  = SimpleTracker(iou_thresh=0.3, max_age=20)

    for vpath in videos:
        cap, fps, W, H = open_video_reader(vpath)
        out_path = out_dir / 'annotated' / (vpath.stem + '_annotated.mp4')
        writer = make_video_writer(str(out_path), fps, W, H)

        frame_id = 0
        t0 = time.time()
        save_idxs = []
        total_dets = 0
        csv_rows = []

        # We'll compute save indices after we know total frames; if not available, we fallback later
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames and n_frames > 0:
            save_idxs = evenly_spaced_indices(n_frames, k=6)

        while True:
            ok, frame = cap.read()
            if not ok: break
            dets = detector.detect_persons(frame)  # (N,5)
            trks = tracker.update(dets)            # (M,6) [x1,y1,x2,y2,score,tid]

            # run pose once per frame; then match poses to tracked boxes by IoU
            poses = poser.estimate_pose(frame)     # list of (box, kpts(17,3), score)

            # Build array for pose boxes to match to track boxes
            pose_boxes = [p[0] for p in poses]
            pose_kpts  = [p[1] for p in poses]
            pose_scores= [p[2] for p in poses]

            annotated = frame.copy()
            total_dets += len(trks)

            # For each track, find the best matching pose by IoU
            def _iou(a,b):
                xA=max(a[0],b[0]);yA=max(a[1],b[1]);xB=min(a[2],b[2]);yB=min(a[3],b[3])
                inter=max(0,xB-xA)*max(0,yB-yA)
                areaA=max(0,a[2]-a[0])*max(0,a[3]-a[1])
                areaB=max(0,b[2]-b[0])*max(0,b[3]-b[1])
                return inter/(areaA+areaB-inter+1e-6)

            for tb in trks:
                box = tb[:4]; score = tb[4]; tid = int(tb[5])
                best_i = -1; best_j = -1
                for j, pb in enumerate(pose_boxes):
                    i = _iou(box, pb)
                    if i > best_i:
                        best_i = i; best_j = j
                draw_box(annotated, box, tid=tid, score=score)

                # Draw pose if found
                if best_j >= 0 and best_i >= 0.3:
                    draw_skeleton(annotated, pose_kpts[best_j])

                    # flatten kpts for CSV
                    flat_k = pose_kpts[best_j].reshape(-1)
                else:
                    flat_k = np.zeros((17*3,), float)

                csv_rows.append([frame_id, tid, *box.tolist(), score, *flat_k.tolist()])

            writer.write(annotated)

            # Save screenshots if needed
            if save_idxs and frame_id in save_idxs:
                ss_path = out_dir / 'screenshots' / f'{vpath.stem}_frame{frame_id:05d}.jpg'
                cv2.imwrite(str(ss_path), annotated)

            frame_id += 1

        writer.release(); cap.release()
        elapsed = time.time() - t0
        fps_rt  = frame_id / max(1e-6, elapsed)

        # If we couldn't precompute frame indices (unknown frame count), sample now
        if not save_idxs:
            save_idxs = [int(i * frame_id/7) for i in range(1,7)]

        # Write CSV
        import pandas as pd
        cols = ['frame','track_id','x1','y1','x2','y2','score'] + [f'k{i}_{c}' for i in range(17) for c in ('x','y','p')]
        df = pd.DataFrame(csv_rows, columns=cols)
        csv_path = out_dir / 'csv' / f'{vpath.stem}_tracks.csv'
        df.to_csv(csv_path, index=False)

        # Write metrics json
        metrics = {
            'video': str(vpath),
            'frames': frame_id,
            'runtime_sec': round(elapsed,3),
            'fps_runtime': round(fps_rt,2),
            'avg_detections_per_frame': round(total_dets / max(1, frame_id), 3),
        }
        (out_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'metrics' / f'{vpath.stem}_metrics.json','w') as f:
            json.dump(metrics, f, indent=2)

        print(f'Done: {vpath.name}  ->  {out_path.name}  |  FPS: {fps_rt:.2f}')

if __name__ == '__main__':
    main()
