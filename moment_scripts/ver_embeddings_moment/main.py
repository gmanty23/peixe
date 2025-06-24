from dataset import load_npz_files_classified, create_dataloader
from model_setup import load_moment_embedding_model
from embed import get_embeddings
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os

def main():
    data_dir = "/home/gmanty/code/AnemoNAS/moment/clases/"
    channel_indices = [34,35,36,37,38,39,40,41]
    batch_size = 64

    print("Cargando datos...")
    X, Y, file_list = load_npz_files_classified(data_dir, channel_indices)

    print("Dividiendo en train/test...")
    X_train, X_test, Y_train, Y_test, files_train, files_test = train_test_split(
        X, Y, file_list, test_size=0.2, stratify=Y, random_state=42
    )

    train_loader = create_dataloader(X_train, Y_train, batch_size)
    test_loader = create_dataloader(X_test, Y_test, batch_size)

    print("Cargando modelo MOMENT...")
    model = load_moment_embedding_model()

    print("Extrayendo embeddings train...")
    train_embeddings, train_labels, train_paths = get_embeddings(model, train_loader, files_train)
    print("Extrayendo embeddings test...")
    test_embeddings, test_labels, test_paths = get_embeddings(model, test_loader, files_test)

    canales_str = "_".join(str(c) for c in channel_indices)
    save_dir = f"canales_{canales_str}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "embeddings.npz")

    np.savez(save_path,
             embeddings=np.vstack([train_embeddings, test_embeddings]),
             labels=np.concatenate([train_labels, test_labels]),
             paths=np.concatenate([train_paths, test_paths]))
    print(f"Embeddings guardados en {save_path}")

if __name__ == "__main__":
    main()
