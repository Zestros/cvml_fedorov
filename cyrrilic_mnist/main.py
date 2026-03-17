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
