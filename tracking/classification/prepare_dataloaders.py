import os
import random
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from blob_dataset import BlobDataset  
from sklearn.model_selection import StratifiedShuffleSplit

# Función para preparar los dataloaders
# Esta función carga el dataset de blobs y lo divide en conjuntos de entrenamiento y validación
# según la proporción especificada. También aplica transformaciones a las imágenes.
#
# Parámetros:
# - root_dir: directorio raíz donde se encuentran las imágenes de los blobs.
# - csv_path: ruta al archivo CSV que contiene las etiquetas y coordenadas de los blobs.    
# - input_size: tamaño de entrada para las imágenes (se recortan a este tamaño).
# - batch_size: tamaño del lote para los dataloaders.
# - val_ratio: proporción del conjunto de datos que se utilizará para la validación.
# - seed: semilla para la aleatoriedad (para reproducibilidad).
def prepare_dataloaders(root_dir="tracking/classification/dataset_blobs", csv_path="tracking/classification/dataset_blobs/labels.csv", input_size=256, batch_size=32, val_ratio=0.2, seed=42):
    
    # Transformaciones para el dataset 
    # Además de redimensionar y normalizar, se aplican otras variaciones para aumentar la diversidad del dataset ( 'data augmentation').
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(),                               # voltear horizontalmente
        T.RandomRotation(degrees=15),                           # rotar aleatoriamente            
        T.ColorJitter(brightness=0.01, contrast=0.01),            # variaciones de brillo y contraste
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] → [-1,1]
    ])

    dataset = BlobDataset(root_dir, csv_path, input_size=input_size, transform=transform)

    labels = dataset.targets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(splitter.split(X=range(len(dataset)), y=labels))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader