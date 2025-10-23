import cv2
import numpy as np

# COCO keypoint skeleton connections (18->common subset) for nice-looking bones
# YOLOv8-Pose returns 17 kpts: [nose, eyeL, eyeR, earL, earR, shoulderL, shoulderR, elbowL, elbowR,
#                               wristL, wristR, hipL, hipR, kneeL, kneeR, ankleL, ankleR]
COCO_EDGES = [
    (5,7),(7,9),      # left arm
    (6,8),(8,10),     # right arm
    (11,13),(13,15),  # left leg
    (12,14),(14,16),  # right leg
    (5,6),            # shoulders
    (11,12),          # hips
    (5,11),(6,12),    # torso diagonals
]

def draw_box(img, box, color=(0,255,0), tid=None, score=None):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    label = []
    if tid is not None: label.append(f'ID {tid}')
    if score is not None: label.append(f'{score:.2f}')
    if label:
        txt = ' | '.join(label)
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1,y1-18), (x1+tw+6, y1), color, -1)
        cv2.putText(img, txt, (x1+3,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def draw_skeleton(img, kpts, conf=None, color=(255,255,255)):
    # kpts: array of shape (17,2) or (17,3) where last is score
    pts = np.array(kpts, dtype=float)
    if pts.shape[1] == 3:
        xy = pts[:,:2]; sc = pts[:,2]
    else:
        xy = pts; sc = np.ones((pts.shape[0],), float)

    for i,j in COCO_EDGES:
        if sc[i] > 0 and sc[j] > 0:
            p1 = tuple(map(int, xy[i])); p2 = tuple(map(int, xy[j]))
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

    for i, (x,y) in enumerate(xy.astype(int)):
        if sc[i] > 0:
            cv2.circle(img, (x,y), 3, color, -1, lineType=cv2.LINE_AA)
