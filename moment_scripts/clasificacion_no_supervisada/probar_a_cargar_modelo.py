from momentfm import MOMENTPipeline

# Cargar modelo preentrenado en modo embedding
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={'task_name': 'embedding'}
)

# Inicializamos el modelo
model.init()
print(model)

# Enviamos a GPU y establecemos el tipo de dato
import torch
model.to("cuda").float()
