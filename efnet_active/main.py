import cv2
import torch
import time
from common import build_model, transform, device

model = build_model(pretrained=False)
model.eval()

cap = cv2.VideoCapture(1)
while True:
    _, frame = cap.read()
    
    # Predict
    img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.sigmoid(out).item()
    
    label = "PERSON" if prob > 0.5 else "EMPTY"
    color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
    
    cv2.putText(frame, f"{label} ({prob:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
