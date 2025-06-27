from moment_scripts.clasificacion_supervisada.dataset import load_dataset
from moment_scripts.clasificacion_supervisada.model_setup import create_moment_model
from moment_scripts.clasificacion_supervisada.train import train_moment


def main():
    root_dirs = {
        0: "/home/gmanty/code/AnemoNAS/moment/clases/activos/",
        1: "/home/gmanty/code/AnemoNAS/moment/clases/alterados/",
        2: "/home/gmanty/code/AnemoNAS/moment/clases/relajados/"
    }
    channels_to_use = [34, 35, 36, 37, 38, 39, 40, 41]
    train_loader, val_loader, class_weights, val_dataset = load_dataset(root_dirs, channels_to_use)
    model = create_moment_model(n_channels=len(channels_to_use), num_class=3)
    train_moment(model, train_loader, val_loader, class_weights, val_dataset)

if __name__ == "__main__":
    main()
    
    

