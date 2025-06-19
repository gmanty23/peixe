import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter

class NPZDataset(Dataset):
    def __init__(self, file_paths, labels, channels_to_use):
        self.file_paths = file_paths
        self.labels = labels
        self.channels_to_use = channels_to_use

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz = np.load(self.file_paths[idx])
        data = npz["data"][self.channels_to_use, :]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor

def load_dataset(root_dirs, channels_to_use, val_split=0.2, batch_size=64):
    all_files = []
    all_labels = []
    for label, dir_path in root_dirs.items():
        npz_files = glob.glob(os.path.join(dir_path, "*.npz"))
        all_files.extend(npz_files)
        all_labels.extend([label] * len(npz_files))

    dataset = NPZDataset(all_files, all_labels, channels_to_use)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    weights = []
    for i in range(len(label_counts)):
        weights.append(total / (label_counts[i] + 1e-6))
    weights = torch.tensor(weights, dtype=torch.float32)

    return train_loader, val_loader, weights, val_dataset
