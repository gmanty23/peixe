from momentfm import MOMENTPipeline
import torch

def load_moment_embedding_model():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'embedding'}
    )
    model.init()
    model.to("cuda").float()
    return model
