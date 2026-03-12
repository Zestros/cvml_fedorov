#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


transform_train = transforms.Compose([
    transforms.Resize((32, 32)), #transforms.Resize((256, 256)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mydata = MyDataset(Path.cwd() / "Cyrillic")
train, test = mydata.split_train_test(0.2, transform_train, transform_test)

batch_size = 64
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
"""
train_loader_print = DataLoader(train, batch_size=8, shuffle=True)
images, labels = next(iter(train_loader_print))

plt.figure(figsize=(15, 5))
for i in range(len(images)):
    plt.subplot(1, 8, i + 1)

    img = images[i]

    plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')

    class_name = train.classes[labels[i]]
    plt.title(class_name)

plt.show()
"""

save_path = Path.cwd()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model = RusMNIST().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params=}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 15
train_loss = []
train_acc = []

model_path = save_path / "model.pth"
if not model_path.exists():
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = (images.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        epoch_loss = run_loss / len(train_loader)
        epoch_acc = 100 * (correct / total)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"{epoch}, {epoch_loss:=.3f}, {epoch_acc:=.3f}")
    torch.save(model.state_dict(), model_path)

    plt.figure()
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss)
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(train_acc)
    plt.show()

else:
    model.load_state_dict(torch.load(model_path))
