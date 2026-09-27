"""
model.py
ResNet18 preentrenada en ImageNet, con la mayoría de capas congeladas (transfer
learning) y una nueva cabeza de clasificación binaria. Congelar el backbone tiene
sentido aquí porque el dataset es pequeño (~5.800 imágenes): entrenar la red entera
desde cero sobrejustaría rápido.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Reemplazamos la última capa (fc); esta sí es entrenable siempre
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
