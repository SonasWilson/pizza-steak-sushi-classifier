import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from pathlib import Path



def create_model(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last block only
    for param in model.layer4.parameters():
        param.requires_grad = True

    #

    # add dropout and fc layer
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), # probability of 30%
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model