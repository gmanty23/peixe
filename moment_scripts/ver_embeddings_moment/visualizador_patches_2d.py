import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import os

def visualizar_embeddings_patches(npz_path):
    # Cargar datos
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    paths = data["paths"]

    print(f"Embeddings shape original: {emb.shape}")
    
    if emb.ndim != 3:
        raise ValueError(f"Este visualizador espera embeddings con 3 dimensiones (secuencia, patch, feature). Tu array tiene {emb.ndim} dimensiones.")
    
    n_seqs, n_patches, n_features = emb.shape

    # Aplanar: cada patch es un punto
    emb_flat = emb.reshape(-1, n_features)  # (n_seqs * n_patches, n_features)

    # Repetir nombres de archivo
    paths_repeated = np.repeat(paths, n_patches)

    # Generar índices de patch
    patch_indices = np.tile(np.arange(n_patches), n_seqs)

    # Aplicar PCA
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb_flat)
    print(f"PCA result shape: {emb_pca.shape}")

    # Dibujar
    fig = px.scatter(
        x=emb_pca[:, 0],
        y=emb_pca[:, 1],
        hover_data={"Archivo": paths_repeated, "Patch": patch_indices}
    )

    # Guardar y abrir
    html_path = "pca_patches.html"
    fig.write_html(html_path)
    print(f"Plot guardado en {html_path}")
    os.system(f"explorer.exe {html_path}")
