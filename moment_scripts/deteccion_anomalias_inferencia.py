from moment.momentfm import MOMENTPipeline
from pprint import pprint
import torch
import numpy as np
from tqdm import tqdm
import os


# Cargamos el modelo MOMENT en reconstrucción para detección de anomalías
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "reconstruction"},  # Modo usado para detección de anomalías
    # local_files_only=True,  # (opcional) si ya tienes el modelo descargado
)

# Inicialización e inspección del modelo
model.init()
print(model)

# Carga de datos de entrada, recolectando y concatenando los bloques de datos
input_path = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/moment_inputs"
output_path = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/moment_outputs"
if not os.path.exists(output_path):
    os.makedirs(output_path)

X_all = []
for filename in sorted(os.listdir(input_path)):
    if filename.endswith(".npz"):
        data = np.load(os.path.join(input_path, filename))
        X = data["data"]  # [N, 512, D]
        X_all.append(X)
    
X = np.concatenate(X_all, axis=0)  # [N_total, 512, D]
X = np.transpose(X, (0, 2, 1))     # [N_total, D, 512]
x_tensor = torch.tensor(X, dtype=torch.float32)

# Realizamos la inferencia para detección de anomalías
model = model.to("cuda").float()  # Aseguramos que el modelo esté en la GPU
model.eval()  # Modo evaluación

batch_size = 8
trues, preds = [], []

with torch.no_grad():
    for i in tqdm(range(0, x_tensor.shape[0], batch_size)):
        batch = x_tensor[i:i+batch_size].to("cuda").float()
        output = model(x_enc=batch)
        trues.append(batch.cpu().numpy())
        preds.append(output.reconstruction.cpu().numpy())

trues = np.concatenate(trues, axis=0)  # [N_total, D, 512]
preds = np.concatenate(preds, axis=0)  # [N_total, D, 512]

# Calcular error MSE por ventana: promedio por canal y tiempo
errors = ((trues - preds) ** 2).mean(axis=(1, 2))  # [N_total]

# Guardar salida global con errores incluidos
np.savez(output_path, trues=trues, preds=preds, errors=errors)