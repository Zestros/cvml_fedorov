import cv2
import torch
import torch.nn as nn
from collections import deque
from common import build_model, transform, model_path, device

class Buffer():
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)
    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)
    def get_batch(self):
        return torch.stack(list(self.frames)).to(device), torch.tensor(list(self.labels), dtype=torch.float32).to(device)

model = build_model(pretrained=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
buffer = Buffer()
count_labeled = 0

cap = cv2.VideoCapture(1)
while True:
    _, frame = cap.read()
    cv2.imshow("Training Mode", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"): break
    if key in [ord("1"), ord("2")]:
        label = 1.0 if key == ord("1") else 0.0
        buffer.append(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), label)
        count_labeled += 1

    if count_labeled >= 16:
        model.train()
        images, labels = buffer.get_batch()
        optimizer.zero_grad()
        loss = criterion(model(images).view(-1), labels)
        loss.backward()
        optimizer.step()
        print(f"Trained! Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), model_path) # Автосохранение
        count_labeled = 0

cap.release()
cv2.destroyAllWindows()
