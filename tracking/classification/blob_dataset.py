import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class BlobDataset(Dataset):
    def __init__(self, root_dir, csv_path, input_size=256, transform=None):
        # Mapeo de etiquetas a enteros
        self.label_map = {
            'individual': 0,
            'group': 1,
            'ruido': 2,
            #'reflejo': 3
        }
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_path)
        self.targets = [self.label_map[clase] for clase in self.data['class']]
        self.input_size = input_size
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nombre, clase, cx, cy = self.data.iloc[idx]
        cx, cy = int(cx), int(cy)
        label = self.label_map[clase]

        # Ruta completa a la imagen
        path = os.path.join(self.root_dir, clase, nombre)
        img = Image.open(path).convert("RGB")
        w, h = img.size

        # Recorte centrado en (cx, cy)
        half = self.input_size // 2
        left = max(cx - half, 0)
        upper = max(cy - half, 0)
        right = min(left + self.input_size, w)
        lower = min(upper + self.input_size, h)
        img = img.crop((left, upper, right, lower))

        # Transformaciones
        if self.transform:
            img = self.transform(img)

        return img, label
