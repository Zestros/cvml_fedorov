#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("/Users/zestros/my_prog/my_python/runs/detect/figures/yolo4/weights/best.pt")
model.to("mps") 

camera = cv2.VideoCapture(1)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    t_start = time.perf_counter()

    results = model.predict(frame, conf=0.5, verbose=False)

    dt = time.perf_counter() - t_start
    fps = 1 / dt if dt > 0 else 0

    result = results[0]
    annotator = Annotator(frame)

    if result.boxes is not None:
        for box in result.boxes:
            b = box.xyxy[0] 
            cls = int(box.cls[0])
            conf = box.conf[0]

            label = f"{model.names[cls]} {conf:.2f}"
            annotator.box_label(b, label, color=(0, 255, 0))

    annotated_frame = annotator.result()

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


# In[ ]:




