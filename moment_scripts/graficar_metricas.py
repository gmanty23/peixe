import os
import json
import matplotlib.pyplot as plt
from natsort import natsorted

# Define la carpeta de trabajo (modifica esta variable con la ruta que necesites)
carpeta_trabajo = "/home/gmanty/code/AnemoNAS/outputs_moment/"

# Recorre las subcarpetas de la carpeta de trabajo
for subcarpeta in os.listdir(carpeta_trabajo):
    subcarpeta_path = os.path.join(carpeta_trabajo, subcarpeta)
    if os.path.isdir(subcarpeta_path):
        # Busca las carpetas metrics_* dentro de esta subcarpeta
        metrics_dirs = [d for d in os.listdir(subcarpeta_path) if d.startswith("metrics_")]
        metrics_dirs = natsorted(metrics_dirs)

        all_epochs = []
        all_train_loss = []
        all_val_loss = []
        all_train_acc = []
        all_val_acc = []

        current_epoch_offset = 0

        for metrics_dir in metrics_dirs:
            history_path = os.path.join(subcarpeta_path, metrics_dir, "history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)

                for entry in history:
                    all_epochs.append(entry["epoch"] + current_epoch_offset)
                    all_train_loss.append(entry["train_loss"])
                    all_val_loss.append(entry["val_loss"])
                    all_train_acc.append(entry["train_acc"])
                    all_val_acc.append(entry["val_acc"])

                # Actualiza el offset
                current_epoch_offset = all_epochs[-1]

        if all_epochs:
            # Crear figura con 2x2 subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            # Train Loss
            axs[0, 0].plot(all_epochs, all_train_loss, marker='o', color='blue')
            axs[0, 0].set_title("Train Loss over Epochs")
            axs[0, 0].set_xlabel("Epoch")
            axs[0, 0].set_ylabel("Train Loss")
            axs[0, 0].grid(True)

            # Validation Loss
            axs[0, 1].plot(all_epochs, all_val_loss, marker='o', color='red')
            axs[0, 1].set_title("Validation Loss over Epochs")
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("Validation Loss")
            axs[0, 1].grid(True)

            # Train Accuracy
            axs[1, 0].plot(all_epochs, all_train_acc, marker='o', color='green')
            axs[1, 0].set_title("Train Accuracy over Epochs")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Train Accuracy")
            axs[1, 0].grid(True)

            # Validation Accuracy
            axs[1, 1].plot(all_epochs, all_val_acc, marker='o', color='orange')
            axs[1, 1].set_title("Validation Accuracy over Epochs")
            axs[1, 1].set_xlabel("Epoch")
            axs[1, 1].set_ylabel("Validation Accuracy")
            axs[1, 1].grid(True)

            plt.tight_layout()

            # Guardar figura
            output_file = f"{subcarpeta}_metrics.png"
            output_path = os.path.join(subcarpeta_path, output_file)
            plt.savefig(output_path)
            plt.close()

            print(f"Gráfica guardada: {output_path}")
        else:
            print(f"No se encontraron métricas en: {subcarpeta_path}")
