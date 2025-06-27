from momentfm import MOMENTPipeline
import torch

def create_moment_model(n_channels, num_class):
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            'n_channels': n_channels,
            'num_class': num_class
        }
    )
    model.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).float()
    return model
