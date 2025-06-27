import sys
import json
from train_interfaz import train_moment
from dataset_interfaz import load_dataset
from model_setup_interfaz import create_moment_model
import torch

def main():
    # Lee parámetros del JSON pasado como argumento
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = json.load(f)

    input_dirs = config["input_dirs"]
    channels_to_use = config["channels_to_use"]
    params = config["params"]
    output_dir = config["output_dir"]
    checkpoint_to_load = config.get("checkpoint_to_load")

    train_loader, val_loader, class_weights, val_dataset = load_dataset(
        input_dirs, channels_to_use,
        val_split=0.2,
        batch_size=params["batch_size"]
    )

    model = create_moment_model(n_channels=len(channels_to_use), num_class=3)
    if checkpoint_to_load:
        model.load_state_dict(torch.load(checkpoint_to_load, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    train_moment(model, train_loader, val_loader, class_weights, val_dataset, params, output_dir, print)

if __name__ == "__main__":
    main()
