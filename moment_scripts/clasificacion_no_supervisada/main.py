from dataset import load_npz_files_classified, create_dataloader
from model_setup import load_moment_embedding_model
from embed import get_embeddings
from train_svm import train_svm
import torch

import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # CONFIGURACIÓN
    data_dir = "/home/gmanty/code/AnemoNAS/moment/clases/" 
    channel_indices = [0,1,2,3,4,5,6,7]
    batch_size = 64
    class_names = ["activos", "alterados", "relajados"]

    # 1️⃣ Cargar datos y etiquetas
    print("Cargando datos...")
    X_tensor, Y_tensor = load_npz_files_classified(data_dir, channel_indices)
    X_np = X_tensor.numpy()
    Y_np = Y_tensor.numpy()

    # 2️⃣ Separar en train y test
    print("Dividiendo en train/test...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_np, Y_np, test_size=0.2, stratify=Y_np, random_state=42
    )

    # 3️⃣ Crear DataLoaders
    train_loader = create_dataloader(torch.tensor(X_train), torch.tensor(Y_train), batch_size)
    test_loader = create_dataloader(torch.tensor(X_test), torch.tensor(Y_test), batch_size)

    # 4️⃣ Cargar MOMENT
    print("Cargando modelo MOMENT...")
    model = load_moment_embedding_model()

    # 5️⃣ Extraer embeddings
    print("Extrayendo embeddings train...")
    train_embeddings, train_labels = get_embeddings(model, train_loader)
    print("Extrayendo embeddings test...")
    test_embeddings, test_labels = get_embeddings(model, test_loader)

    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")



    # 6️⃣ Entrenar SVM y guardar resultados
    canales_str = "_".join(str(c) for c in channel_indices)
    save_dir = f"canales_{canales_str}"
    save_prefix = "clases_" + "_".join(str(c) for c in sorted(set(Y_np)))

    train_svm(train_embeddings, train_labels, test_embeddings, test_labels, 
            class_names=class_names, save_dir=save_dir, save_prefix=save_prefix)

if __name__ == "__main__":
    main()
