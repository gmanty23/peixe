from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

def visualizar_embeddings_3d(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    labels = data["labels"]
    paths = data["paths"]

    if emb.ndim > 2:
        emb = emb.mean(axis=1)  # O emb = emb.reshape(emb.shape[0], -1)


    pca = PCA(n_components=3)
    emb_pca = pca.fit_transform(emb)

    fig = px.scatter_3d(
        x=emb_pca[:, 0], y=emb_pca[:, 1], z=emb_pca[:, 2],
        color=labels.astype(str),
        hover_data={"Archivo": paths, "Clase": labels}
    )
    fig.write_html("pca_plot_3d.html")
    print("Plot guardado como pca_plot_3d.html")
    import os
    os.system("explorer.exe pca_plot_3d.html")