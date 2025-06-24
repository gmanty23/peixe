import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

def visualizar_embeddings_2d(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    labels = data["labels"]
    paths = data["paths"]

    if emb.ndim > 2:
        emb = emb.mean(axis=1)  # O emb = emb.reshape(emb.shape[0], -1)


    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb)

    fig = px.scatter(
        x=emb_pca[:,0], y=emb_pca[:,1],
        color=labels.astype(str),
        hover_data={"Archivo": paths, "Clase": labels}
    )
    fig.write_html("pca_plot.html")
    print("Plot guardado como pca_plot.html")

    # Si estás en WSL, abre con explorer.exe
    import os
    os.system("explorer.exe pca_plot.html")
