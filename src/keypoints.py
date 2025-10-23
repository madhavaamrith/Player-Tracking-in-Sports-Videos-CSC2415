from ultralytics import YOLO
import numpy as np
import cv2
import torch

def _resolve_device(dev):
    if dev is None or str(dev).lower() in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(dev)

class YoloPoseEstimator:
    def __init__(self, model_name='yolov8n-pose.pt', device='auto', conf=0.25):
        self.device = _resolve_device(device)
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf = conf

    def estimate_pose(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=rgb,
            conf=self.conf,
            verbose=False,
            device=self.device
        )
        kpt_list = []
        if len(results):
            r = results[0]
            if r.boxes is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                kpts = r.keypoints.data.cpu().numpy()  # (N,17,3)
                for b, s, k in zip(boxes, scores, kpts):
                    kpt_list.append((b, k, float(s)))
        return kpt_list
