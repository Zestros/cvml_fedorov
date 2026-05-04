#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import random 
import string

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, n=200, size=128, mode=1):
        super().__init__()
        self.n = n
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        image = Image.new("L", (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        text = "ABC"
        x, y = 30, 30
        if self.mode == 1:
            x, y = random.randint(10, self.size-40), random.randint(10, self.size-40)

        elif self.mode == 2:
            text = ''.join(random.choices(string.ascii_uppercase, k=3))

        elif self.mode == 3:
            text = ''.join(random.choices(string.ascii_uppercase, k=random.randint(1, 4)))

        elif self.mode == 4:
            text = ''.join(random.choices(string.ascii_uppercase, k=random.randint(1, 4)))
            x, y = random.randint(10, self.size-40), random.randint(10, self.size-40)

        draw.text((x,y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor

ds = ImageDataset()
image = ds[0]

plt.imshow(ds[0][0][0])
plt.show()

class Encoder(nn.Module):
    def __init__(self, latent = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent = 512):
        super().__init__()
        self.bottleneck = nn.Linear(latent, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x

if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print(f"Параметров в Encoder: {count_parameters(encoder):,}")
    print(f"Параметров в Decoder: {count_parameters(decoder):,}")


    dataset = ImageDataset(2000,256,4)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder.to(device)
    decoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    encoder.train()
    decoder.train()

    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            latent = encoder(imgs)
            output = decoder(latent)
            loss = criterion(imgs, output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"{epoch=}, {avg_loss=:.2f}")

    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")


# In[11]:


#from encoder_decoder_train import (Decoder, Encoder, ImageDataset)
import torch
import matplotlib.pyplot as plt

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))

encoder.to(device)
decoder.to(device)

encoder.eval()
decoder.eval()

dataset = ImageDataset(2000,256,4)
image, _ = dataset[0]

with torch.no_grad():
    latent = encoder(image.unsqueeze(0).to(device)) 
    result = decoder(latent)

plt.subplot(131)
plt.imshow(image.squeeze().detach().cpu())
plt.subplot(132)
plt.imshow(result.squeeze().detach().cpu())
plt.subplot(133)
plt.imshow(image.squeeze().detach().cpu() - result.squeeze().detach().cpu())
plt.show()


# In[ ]:




