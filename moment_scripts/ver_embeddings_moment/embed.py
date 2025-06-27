import numpy as np
import torch
from tqdm import tqdm

def get_embeddings(model, dataloader, file_list):
    embeddings = []
    labels = []
    paths = []

    model.eval()
    with torch.no_grad():
        idx = 0
        for batch_x, batch_mask, batch_y in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to("cuda").float()
            batch_mask = batch_mask.to("cuda")

            output = model(x_enc=batch_x, input_mask=batch_mask)
            emb = output.embeddings.cpu().numpy()

            embeddings.append(emb)
            labels.append(batch_y.numpy())
            paths.extend(file_list[idx: idx + emb.shape[0]])
            idx += emb.shape[0]

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    paths = np.array(paths)
    return embeddings, labels, paths
