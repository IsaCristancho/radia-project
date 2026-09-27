"""
train.py
Entrena el clasificador de neumonía y evalúa en el set de test con métricas clínicas.

Ejecutar en Kaggle: ajustar DATA_DIR a
    /kaggle/input/chest-xray-pneumonia/chest_xray
"""

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from dataset import get_dataloaders
from model import build_model

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------
DATA_DIR = "/Users/isabe/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"  # local: tras `kaggle datasets download ... --unzip`
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
SEED = 42


def get_device():
    """
    Detecta la mejor GPU disponible: MPS en Apple Silicon (M1-M5), CUDA en
    máquinas con GPU NVIDIA, o CPU como fallback.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Usando dispositivo: {DEVICE}")

# --------------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, classes, criterion=None):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        if criterion is not None:
            running_loss += criterion(outputs, labels).item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]  # prob de clase PNEUMONIA
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset) if criterion is not None else None

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall clase positiva
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nConfusion matrix (filas=real, cols=predicho) — clases: {classes}")
    print(cm)
    print(f"Sensibilidad (recall PNEUMONIA): {sensitivity:.3f}")
    print(f"Especificidad (recall NORMAL):   {specificity:.3f}")
    print(f"AUC-ROC:                         {auc:.3f}")
    print("\n" + classification_report(all_labels, all_preds, target_names=classes))

    return {"loss": avg_loss, "auc": auc, "sensitivity": sensitivity, "specificity": specificity}


def main():
    train_loader, val_loader, test_loader, class_weights, classes = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE
    )
    print(f"Clases: {classes} | Pesos de clase: {class_weights}")

    model = build_model(num_classes=len(classes), freeze_backbone=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    # Solo optimizamos los parámetros entrenables (la cabeza fc, ya que el resto está congelado)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"\n=== Epoch {epoch}/{EPOCHS} — train loss: {train_loss:.4f} ===")
        val_metrics = evaluate(model, val_loader, classes, criterion)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), "model.pt")
            print(f"Nuevo mejor modelo guardado (AUC val: {best_auc:.3f})")

    print("\n\n===== Evaluación final en TEST =====")
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE, weights_only=True))
    evaluate(model, test_loader, classes)


if __name__ == "__main__":
    main()
