import numpy as np
import torch
from tqdm import tqdm

def get_embeddings(model, dataloader):
    """
    Obtiene los embeddings de los datos pasados por el DataLoader.
    
    Args:
        model: MOMENT en modo embedding.
        dataloader: DataLoader con los datos.
        
    Returns:
        embeddings: np.array [N, D] donde D=1024 (embedding size de MOMENT)
    """
    embeddings = []
    labels = []  # aunque los labels sean dummy, por estructura

    model.eval()
    with torch.no_grad():
        for batch_x, batch_mask, batch_y in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to("cuda").float()
            batch_mask = batch_mask.to("cuda")

            output = model(x_enc=batch_x, input_mask=batch_mask)
            emb = output.embeddings.mean(dim=1).cpu().numpy()
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)  # Aseguramos que siempre es [1, 1024] aunque sea un solo sample
            embeddings.append(emb)
            
            labels.append(batch_y.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels
