import cv2
import torch
from super_gradients.training import models
import math
import numpy as np
from numpy import random
from sort import Sort
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")   
model = models.get('yolo_nas_s', pretrained_weights='coco').to(device)
cap = cv2.VideoCapture('test.mov')
assert cap.isOpened(), "Error reading video file"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output/test_output.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (frame_width, frame_height))
person_class_index = 0
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.1)
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        cv2.putText(frame, f'Frame: {count}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        detections = np.empty((0,6))
        result = model.predict(frame, conf=0.5, fuse_model=False)
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            if cls == person_class_index: 
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = map(int, bbox)
                conf = math.ceil((confidence*100))/100
                label = f'person {conf}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color=(0, 0, 255), thickness=-1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], thickness=1)
                detections = np.vstack((detections, [x1, y1, x2, y2, conf, 0]))
        trackers = tracker.update(detections)
        cv2.putText(frame, f'People Counting: {len(detections)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    else:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
