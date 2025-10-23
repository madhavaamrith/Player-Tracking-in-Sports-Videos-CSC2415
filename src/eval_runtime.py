# src/eval_runtime.py
import json, glob, os, pathlib
from collections import defaultdict

def gather(run_dir):
    rows = []
    for p in pathlib.Path(run_dir, "metrics").glob("*.json"):
        with open(p, "r") as f:
            m = json.load(f)
        rows.append({
            "run": pathlib.Path(run_dir).name,
            "video": pathlib.Path(m["video"]).stem,
            "frames": m["frames"],
            "fps_runtime": m["fps_runtime"],
            "avg_det_per_frame": m["avg_detections_per_frame"],
        })
    return rows

def main():
    runs = []
    for d in sorted(glob.glob("outputs_*")) + ["outputs",]:
        if os.path.isdir(d) and os.path.isdir(os.path.join(d,"metrics")):
            runs += gather(d)
    if not runs:
        print("No metrics found.")
        return

    # Pretty print
    vids = sorted({r["video"] for r in runs})
    run_names = sorted({r["run"] for r in runs})
    print(f"{'Video':30}", end="")
    for rn in run_names: print(f"{rn:>20}", end="")
    print()

    def find(rn, v, key):
        for r in runs:
            if r["run"]==rn and r["video"]==v:
                return r[key]
        return None

    for v in vids:
        print(f"{v:30}", end="")
        for rn in run_names:
            fps = find(rn, v, "fps_runtime")
            det = find(rn, v, "avg_det_per_frame")
            if fps is None:
                print(f"{'—':>20}", end="")
            else:
                print(f"{fps:.1f} fps / {det:.2f} det", end=" "*(20 - len(f"{fps:.1f} fps / {det:.2f} det")))
        print()

if __name__ == "__main__":
    main()
