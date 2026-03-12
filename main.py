import torch.optim as optim
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import copy
from collections import defaultdict
from torch.utils.data import DataLoader
from model import RusMNIST, MyDataset

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mydata = MyDataset(Path.cwd() / "Cyrillic")
train, test = mydata.split_train_test(0.2, None, transform_test)

batch_size = 64
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

save_path = Path.cwd()
model_path = save_path / "model.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model = RusMNIST().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

it = iter(test_loader)
images, labels = next(it)
image = images[0].unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output,1)
print(f"True - {labels[0]}")
print(f"Pred - {predicted.cpu().item()}")



model.load_state_dict(torch.load(model_path))
model.eval()
total = 0
correct = 0 

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = (images.to(device), labels.to(device))   
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:=.3f}")
