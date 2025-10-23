from ultralytics import YOLO
import numpy as np
import torch

def _resolve_device(dev):
    if dev is None or str(dev).lower() in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(dev)

class YoloDetector:
    def __init__(self, model_name='yolov8n.pt', device='auto', conf=0.25):
        self.device = _resolve_device(device)
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf = conf
        self.person_id = 0  # COCO 'person'

    def detect_persons(self, image_bgr):
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=rgb,
            conf=self.conf,
            classes=[self.person_id],
            verbose=False,
            device=self.device
        )
        dets = []
        if len(results):
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                for xyxy, score, cls in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(boxes))
                ):
                    dets.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(score)])
        return np.array(dets, dtype=float)
