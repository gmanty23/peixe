import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_npz_files_classified(data_dir, channel_indices):
    """
    Carga archivos .npz organizados en subcarpetas de clases y devuelve tensores.
    
    Args:
        data_dir (str): Ruta a la carpeta con las subcarpetas de clases.
        channel_indices (list): Índices de canales a usar (por ejemplo, [0-7]).
        
    Returns:
        X_tensor: [N, 8, 512]
        Y_tensor: [N]
    """
    data_list = []
    label_list = []

    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"Clases detectadas: {class_to_idx}")

    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith('.npz'):
                path = os.path.join(class_dir, fname)
                arr = np.load(path)['data']
                selected = arr[channel_indices, :]
                data_list.append(selected)
                label_list.append(class_idx)

    X = np.stack(data_list)
    Y = np.array(label_list)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    
    return X_tensor, Y_tensor

def create_dataloader(X_tensor, Y_tensor, batch_size=64):
    mask_tensor = torch.ones(X_tensor.shape[0], X_tensor.shape[2], dtype=torch.bool)
    dataset = TensorDataset(X_tensor, mask_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # ✅ drop_last=True
    return dataloader
