import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = Path.cwd() / "model.pth"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_model(pretrained=True):
    # Используем B0, заменяем классификатор на один слой для стабильности на малых данных
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.efficientnet_b0(weights=weights)
    
    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features, 1)
    
    if not pretrained and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model.to(device)
