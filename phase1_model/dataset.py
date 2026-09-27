"""
dataset.py
Carga el dataset Chest X-Ray Pneumonia (estructura de carpetas train/val/test,
cada una con subcarpetas NORMAL/ y PNEUMONIA/) usando torchvision.ImageFolder,
y calcula los pesos de clase para compensar el desbalance.
"""

import os
from collections import Counter

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224

# Transforms: normalización estándar de ImageNet porque usamos un backbone preentrenado
# en ImageNet (ResNet18). Aumentamos datos solo en train.
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # las radiografías son de 1 canal
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    data_dir debe contener subcarpetas train/, val/, test/, cada una con NORMAL/ y
    PNEUMONIA/. Devuelve (train_loader, val_loader, test_loader, class_weights, classes).
    """
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_transforms)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_transforms)

    classes = train_ds.classes  # ['NORMAL', 'PNEUMONIA'] (orden alfabético)

    # Peso de clase inverso a la frecuencia, para que la loss no ignore la clase minoritaria
    counts = Counter([label for _, label in train_ds.samples])
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / (len(counts) * counts[i]) for i in range(len(classes))],
        dtype=torch.float32,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights, classes
