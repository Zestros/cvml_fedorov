#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
from playsound3 import playsound
import time


import os
sound_dir = "wrksnd"
sounds = sorted([os.path.join(sound_dir, f) for f in os.listdir(sound_dir) if f.endswith(('.mp3', '.wav'))])


def get_angle(a,b,c):
    cb = np.atan2(c[1] - b[1],c[0] - b[0])
    ab = np.atan2(a[1] - b[1],a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle


def is_arms_straight(keypoints):
    left_angle = get_angle(keypoints[5], keypoints[7], keypoints[9])
    right_angle = get_angle(keypoints[6], keypoints[8], keypoints[10])

    return left_angle > 160 and right_angle > 160

def is_arms_bent(keypoints):
    left_angle = get_angle(keypoints[5], keypoints[7], keypoints[9])
    right_angle = get_angle(keypoints[6], keypoints[8], keypoints[10])

    return left_angle < 90 and right_angle < 90

ps = None
def detect_pushups(annotated, keypoints, stage, counter):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    if left_shoulder and right_shoulder and left_elbow and right_elbow and left_wrist and right_wrist:
        if is_arms_bent(keypoints) and stage == 0:
            stage = 1
    if left_shoulder and right_shoulder and left_elbow and right_elbow and left_wrist and right_wrist:               
        if is_arms_straight(keypoints) and stage == 1:
            stage = 0
            counter += 1
            return True, stage, counter

    return None, stage, counter


model = YOLO("yolo26n-pose.pt")
model.to("mps")
camera = cv2.VideoCapture(1)

counter = 0
stage = 0 # 0 - сверху, 1 - снизу
while camera.isOpened():
    ret, frame = camera.read()
    #cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    t = time.perf_counter()
    results = model(frame)
    print(f"Elapsed time {(time.perf_counter() - t):.2f}",f"FPS {1 / (time.perf_counter() - t):.1f}")

    if not results:
        continue
    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    visible = True
    for i in [5, 6, 7, 8, 9, 10]:
        conf = result.keypoints.data[0][i][2]
        if conf < 0.5:
            visible = False
            break

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()
    if visible:
        stat, stage, counter = detect_pushups(annotated, keypoints[0], stage, counter)

        if stat:
            if sounds:
                sound_to_play = sounds[(counter - 1) % len(sounds)]
                if ps is None or not ps.is_alive():
                    ps = playsound(sound_to_play, block=False)

    cv2.putText(annotated, f"Count: {counter}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    status = "DOWN" if stage == 1 else "UP"
    cv2.putText(annotated, f"Stage: {status}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Pose", annotated)


# In[ ]:





# In[ ]:




