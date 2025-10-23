import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter/union

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_age=20):
        self.next_id = 1
        self.tracks = {}  # id -> {'box':..., 'age':0, 'miss':0}
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    def update(self, detections):
        # detections: (N,5) [x1,y1,x2,y2,score]
        dets = detections
        assigned = set()
        updates = {}

        # try to match existing tracks
        for tid, t in list(self.tracks.items()):
            best_iou = 0.0; best_j = -1
            for j, d in enumerate(dets):
                if j in assigned: continue
                i = iou(t['box'], d[:4])
                if i > best_iou:
                    best_iou = i; best_j = j
            if best_iou >= self.iou_thresh and best_j >= 0:
                updates[tid] = {'box': dets[best_j][:4], 'score': dets[best_j][4], 'age': t['age']+1, 'miss':0}
                assigned.add(best_j)
            else:
                # no match
                t['miss'] += 1
                t['age']  += 1
                if t['miss'] <= self.max_age:
                    updates[tid] = t

        # create new tracks for unmatched detections
        for j, d in enumerate(dets):
            if j in assigned: continue
            updates[self.next_id] = {'box': d[:4], 'score': d[4], 'age': 1, 'miss': 0}
            self.next_id += 1

        # remove old
        self.tracks = {tid:info for tid,info in updates.items() if info['miss'] <= self.max_age}
        # Pack outputs
        tracked = []
        for tid, info in self.tracks.items():
            x1,y1,x2,y2 = info['box']
            tracked.append([x1,y1,x2,y2, info['score'], tid])
        return np.array(tracked, dtype=float) if len(tracked) else np.zeros((0,6), float)
