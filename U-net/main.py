import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unet_road import UNet, RoadDataset


path = Path("roads")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


ds = RoadDataset(path)

unet = UNet()
unet.load_state_dict(torch.load("unet.pth", map_location=device))
unet.to(device)
unet.eval()


image, mask = ds[8]

with torch.no_grad():
    logits = unet(image.unsqueeze(0).to(device))
    pred = torch.sigmoid(logits)

image_np = image.permute(1, 2, 0).detach().cpu().numpy()
mask_np = mask.squeeze().detach().cpu().numpy()
pred_np = pred.squeeze().detach().cpu().numpy()

pred_bin = (pred_np > 0.5).astype(np.float32)

diff = mask_np - pred_bin

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_np)
plt.title("Исходное изображение")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask_np, cmap="gray")
plt.title("Исходная маска")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(pred_bin, cmap="gray")
plt.title("Предсказанная маска")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(diff)
plt.title("Разница")
plt.axis("off")

plt.tight_layout()
plt.show()
