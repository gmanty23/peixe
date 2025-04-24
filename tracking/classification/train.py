import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from model import BlobCNN
from prepare_dataloaders import prepare_dataloaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# === CONFIGURACIÓN GENERAL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✔️ Usando dispositivo: {device}")
num_epochs = 200
learning_rate = 0.00001
batch_size = 32
input_size = 256

# === RUTAS DE GUARDADO ===
model_dir = "tracking/classification/model"
os.makedirs(model_dir, exist_ok=True)
checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
best_model_path = os.path.join(model_dir, "mejor_modelo.pt")

# === GENERAR CARPETA DE EJECUCIÓN ===
def crear_nueva_run(base_dir="tracking/classification/training_metrics"):
    os.makedirs(base_dir, exist_ok=True)
    existentes = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    indices = [int(d.split("_")[1]) for d in existentes if d.split("_")[1].isdigit()]
    nuevo_id = max(indices) + 1 if indices else 1
    carpeta = os.path.join(base_dir, f"run_{nuevo_id:03d}")
    os.makedirs(carpeta, exist_ok=True)
    return carpeta

run_dir = crear_nueva_run()
metrics_path = os.path.join(run_dir, "metrics.csv")
loss_plot_path = os.path.join(run_dir, "loss_curve.png")
accuracy_plot_path = os.path.join(run_dir, "accuracy_curve.png")
confusion_path = os.path.join(run_dir, "confusion_matrix.png")
per_class_path = os.path.join(run_dir, "per_class_accuracy.csv")
hyperparams_path = os.path.join(run_dir, "hyperparams.json")

# === DATOS ===
train_loader, val_loader = prepare_dataloaders(
    root_dir="tracking/classification/dataset_blobs",
    csv_path="tracking/classification/dataset_blobs/labels.csv",
    input_size=input_size,
    batch_size=batch_size,
)

# === MODELO, OPTIMIZADOR, PÉRDIDA ===
model = BlobCNN(num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12, verbose=True, min_lr=1e-6)

# === CALCULAR PESOS POR CLASE A PARTIR DE LA DISTRIBUCIÓN DEL TRAINING SET ===
# Extraer etiquetas del dataset (ImageFolder asigna train_loader.dataset.targets)
# Acceso a los índices del Subset
subset_indices = train_loader.dataset.indices
# Acceso a las verdaderas etiquetas del dataset base
all_targets = train_loader.dataset.dataset.targets
# Obtener solo las etiquetas del subset de entrenamiento
targets = [all_targets[i] for i in subset_indices]
conteo = Counter(targets)
total = sum(conteo.values())
num_classes = len(conteo)

# Calcular pesos inversamente proporcionales al número de muestras
pesos = [total / conteo[i] for i in range(num_classes)]
pesos_tensor = torch.FloatTensor(pesos).to(device)

# Definir función de pérdida con pesos
criterion = nn.CrossEntropyLoss(weight=pesos_tensor)
print(f"✔️ Pesos por clase aplicados: {pesos}")

# === GUARDAR HIPERPARÁMETROS ===
params = {
    "epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "input_size": input_size,
    "pesos": pesos
}
with open(hyperparams_path, "w") as f:
    json.dump(params, f, indent=4)

start_epoch = 0
best_val_loss = float("inf")
train_losses, val_losses, val_accuracies = [], [], []

# === EARLY STOPPING ===
patience = 75
epochs_sin_mejora = 0

# === CARGAR CHECKPOINT SI EXISTE ===
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    print(f"✔️ Checkpoint cargado (epoch {start_epoch})")

# === ENTRENAMIENTO ===
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_train_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)

    epoch_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # === VALIDACIÓN ===
    model.eval()
    total_val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validación", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_accuracies.append(val_acc)


    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2%}")

    # === GUARDAR MEJOR MODELO ===
    scheduler.step(epoch_val_loss) 

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_sin_mejora = 0
        torch.save(model.state_dict(), best_model_path)
        print("✔️ Nuevo mejor modelo guardado.")
    else:
        epochs_sin_mejora += 1
        print(f"⚠️ {epochs_sin_mejora} época(s) sin mejora en val_loss.")
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            print("🛑 Learning rate mínimo alcanzado. Early stopping activado.")
            break
        if epochs_sin_mejora >= patience:
            print("🛑 Early stopping activado por paciencia.")
            break

    # === GUARDAR CHECKPOINT ===
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, checkpoint_path)

# === GUARDAR MÉTRICAS ===
min_len = min(len(train_losses), len(val_losses), len(val_accuracies))
df = pd.DataFrame({
    "epoch": list(range(start_epoch + 1, start_epoch + 1 + min_len)),
    "train_loss": train_losses[:min_len],
    "val_loss": val_losses[:min_len],
    "val_accuracy": val_accuracies[:min_len]
})
df.to_csv(metrics_path, index=False)

# === GRÁFICO DE PÉRDIDA ===
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolución de la función de pérdida")
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_path)
plt.show()

# === GRÁFICO DE PRECISIÓN ===
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label="Val Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Evolución de la precisión de validación")
plt.legend()
plt.grid(True)
plt.savefig(accuracy_plot_path)
plt.show()

# === MATRIZ DE CONFUSIÓN FINAL ===
classes = ["individual", "group", "ruido"]
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format=".0f")
plt.title("Matriz de confusión (validación final)")
plt.savefig(confusion_path)
plt.show()

# === PRECISIÓN POR CLASE ===

per_class_acc = []
for i, clase in enumerate(classes):
    mask = np.array(all_labels) == i
    correct = np.sum(np.array(all_preds)[mask] == i)
    total = np.sum(mask)
    acc = correct / total if total > 0 else 0
    per_class_acc.append((clase, acc))
    print(f"🔍 {clase}: {acc:.2%} de acierto en validación")

# === GUARDAR PRECISIÓN POR CLASE EN CSV ===
df_acc = pd.DataFrame(per_class_acc, columns=["clase", "accuracy"])
df_acc.to_csv(per_class_path, index=False)

