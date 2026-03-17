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

class MyDataset(Dataset):
    def __init__(self, root, transform=None, seed=42):
        self.seed = seed
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith('.')]) # для .DS_store
        self.class_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_path = self.root / cls_name
            for p in cls_path.glob("*"):
                if p.is_file() and not p.name.startswith('.'): # для .DS_store
                    self.samples.append((p, self.class_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        img = Image.open(path).convert("RGBA")
        image = img.getchannel('A')

        if self.transform:
            image = self.transform(image)

        return image, target

    def split_train_test(self, test_split=0.2, train_transform=None, test_transform=None):

        samples_class = defaultdict(list)
        for path, target in self.samples:
            samples_class[target].append((path, target))

        train_samples, test_samples = [], []
        rng = np.random.default_rng(self.seed)

        for target, class_files in samples_class.items():
            rng.shuffle(class_files)

            split_idx = int(len(class_files) * (1 - test_split))

            if split_idx == 0 and len(class_files) > 0:
                split_idx = 1

            train_samples.extend(class_files[:split_idx])
            test_samples.extend(class_files[split_idx:])

        train_set = copy.deepcopy(self)
        test_set = copy.deepcopy(self)

        rng.shuffle(train_samples)
        rng.shuffle(test_samples)

        train_set.samples = train_samples
        train_set.transform = train_transform or self.transform

        test_set.samples = test_samples
        test_set.transform = test_transform or self.transform

        return train_set, test_set

class RusMNIST(nn.Module):

    def __init__(self):
        super(RusMNIST, self,).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8,512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,34)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu4(x)
        #x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

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
