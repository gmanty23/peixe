import numpy as np
data = np.load("/home/gmanty/code/AnemoNAS/outputs_moment/canales_34_35_36_37_38_39_40_41 (densidad)/embeddings/val_embeddings.npz", allow_pickle=True)
print(data["embeddings"].shape)
print(np.isnan(data["embeddings"]).any())
print(np.all(data["embeddings"] == data["embeddings"][0]))

# from sklearn.decomposition import PCA
# import plotly.express as px

# pca = PCA(n_components=2)
# emb_pca = pca.fit_transform(data["embeddings"])
# print(emb_pca.shape)

# fig = px.scatter(x=emb_pca[:, 0], y=emb_pca[:, 1])
# fig.show()