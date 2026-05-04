from train import (Decoder, Encoder, ImageDataset)
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




