# from momentfm import MOMENTPipeline
# import torch

# # Cambia estos parámetros si quieres más canales o diferente número de clases
# model = MOMENTPipeline.from_pretrained(
#     "AutonLab/MOMENT-1-large",
#     model_kwargs={
#         'task_name': 'classification',
#         'n_channels': 8,    # Número de canales de los datos de entrada
#         'num_class': 3      # Número de clases (modifica según tu problema)
#     }
# )

# model.init()
# # print(model)

# # Prueba de que el modelo funciona con un tensor de ejemplo
# # Simulamos un batch: 16 series, 8 canales, longitud 512
# x_dummy = torch.randn(16, 8, 512)

# # Hacemos un forward pass
# with torch.no_grad():
#     output = model(x_enc=x_dummy)

# # Extraemos los logits y las predicciones
# logits = output.logits
# preds = logits.argmax(dim=1)

# print(f"Logits shape: {logits.shape}")  # [16, 3]
# print(f"Predicciones: {preds}")         # 16 predicciones aleatorias (porque el head está sin entrenar)

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter

# -------- CONFIGURACIÓN --------
root_dirs = {
    0: "/home/gms/AnemoNAS/moment/clases/activos/",  
    1: "/home/gms/AnemoNAS/moment/clases/alterados/",
    2: "/home/gms/AnemoNAS/moment/clases/relajados/"
}

channels_to_use = [0, 1, 2, 3, 4, 5, 6, 7]  # Puedes cambiar dinámicamente

val_split = 0.2
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- DATASET --------
class NPZDataset(Dataset):
    def __init__(self, file_paths, labels, channels_to_use):
        self.file_paths = file_paths
        self.labels = labels
        self.channels_to_use = channels_to_use

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz = np.load(self.file_paths[idx])
        data = npz["data"][self.channels_to_use, :]  # Selecciona canales
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor


# -------- RECOGER ARCHIVOS --------
all_files = []
all_labels = []

for label, dir_path in root_dirs.items():
    npz_files = glob.glob(os.path.join(dir_path, "*.npz"))
    all_files.extend(npz_files)
    all_labels.extend([label] * len(npz_files))

# -------- CREAR DATASET COMPLETO --------
dataset = NPZDataset(all_files, all_labels, channels_to_use)

# -------- DIVIDIR EN TRAIN/VAL --------
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------- CLASS WEIGHTS --------
label_counts = Counter(all_labels)
total = sum(label_counts.values())
weights = []
for i in range(len(label_counts)):
    weights.append(total / (label_counts[i] + 1e-6))  # evitar división por 0

weights = torch.tensor(weights, dtype=torch.float32).to(device)

# Ejemplo de cómo pasar a CrossEntropyLoss:
# criterion = torch.nn.CrossEntropyLoss(weight=weights)

print(f"Train size: {train_size}, Val size: {val_size}")
print(f"Class weights: {weights.cpu().numpy()}")
