import os
import json
import matplotlib.pyplot as plt

# Define la carpeta de trabajo (modifica esta variable con la ruta que necesites)
carpeta = "/home/gmanty/code/AnemoNAS/outputs_moment/canales_34_35_36_37_38_39_40_41/"

# Ruta al archivo history.json
json_path = os.path.join(carpeta, "metrics", "history.json")

# Cargar el JSON
with open(json_path, 'r') as f:
    history = json.load(f)

# Extraer datos
epochs = [entry["epoch"] for entry in history]
train_loss = [entry["train_loss"] for entry in history]
val_loss = [entry["val_loss"] for entry in history]
train_acc = [entry["train_acc"] for entry in history]
val_acc = [entry["val_acc"] for entry in history]

# Crear figura con 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Primer gráfico: Train Loss
axs[0, 0].plot(epochs, train_loss, marker='o', color='blue')
axs[0, 0].set_title("Train Loss over Epochs")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Train Loss")
axs[0, 0].grid(True)

# Segundo gráfico: Validation Loss
axs[0, 1].plot(epochs, val_loss, marker='o', color='red')
axs[0, 1].set_title("Validation Loss over Epochs")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Validation Loss")
axs[0, 1].grid(True)

# Tercer gráfico: Train Accuracy
axs[1, 0].plot(epochs, train_acc, marker='o', color='green')
axs[1, 0].set_title("Train Accuracy over Epochs")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Train Accuracy")
axs[1, 0].grid(True)

# Cuarto gráfico: Validation Accuracy
axs[1, 1].plot(epochs, val_acc, marker='o', color='orange')
axs[1, 1].set_title("Validation Accuracy over Epochs")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Validation Accuracy")
axs[1, 1].grid(True)

# Ajustar diseño
plt.tight_layout()

# Guardar figura
output_path = os.path.join(carpeta, "training_metrics.png")
plt.savefig(output_path)
plt.close()

print(f"Gráfica combinada guardada correctamente en: {output_path}")
