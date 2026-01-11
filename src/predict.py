import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from io import BytesIO
import requests

# Device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['pizza', 'steak', 'sushi']

# model same as training
def create_model(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # add dropout + final layer
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model

# Instantiate model
model = create_model(len(class_names))
model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        pred = torch.argmax(logits, dim=1)
    return class_names[pred.item()]


if __name__ == "__main__":
    url = input("Enter image URL: ")
    print("Predicted class:", predict_image_from_url(url))
