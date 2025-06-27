import os
import json
from natsort import natsorted

# Define la carpeta de trabajo (modifica esta variable con la ruta que necesites)
carpeta_trabajo = "/home/gmanty/code/AnemoNAS/outputs_moment/"

# Diccionario de salida
resultados = {}

# Recorre las subcarpetas de la carpeta de trabajo
for subcarpeta in os.listdir(carpeta_trabajo):
    subcarpeta_path = os.path.join(carpeta_trabajo, subcarpeta)
    if os.path.isdir(subcarpeta_path):
        # Busca las carpetas metrics_* dentro de esta subcarpeta
        metrics_dirs = [d for d in os.listdir(subcarpeta_path) if d.startswith("metrics_")]
        metrics_dirs = natsorted(metrics_dirs)

        all_entries = []
        current_epoch_offset = 0

        for metrics_dir in metrics_dirs:
            history_path = os.path.join(subcarpeta_path, metrics_dir, "history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)

                for entry in history:
                    # Ajusta el número de epoch acumulando offset
                    entry_copy = entry.copy()
                    entry_copy["epoch"] = entry["epoch"] + current_epoch_offset
                    all_entries.append(entry_copy)

                current_epoch_offset = all_entries[-1]["epoch"] + 1  # +1 para la siguiente serie

        if all_entries:
            # Busca el entry con el mejor f1_weighted
            best_entry = max(all_entries, key=lambda x: x.get("f1_weighted", float('-inf')))
            resultados[subcarpeta] = best_entry
            print(f"Mejor epoch en {subcarpeta}: {best_entry['epoch']} con f1_weighted={best_entry.get('f1_weighted')}")
        else:
            print(f"No se encontraron métricas en: {subcarpeta_path}")

# Guarda el JSON con todos los resultados
output_json_path = os.path.join(carpeta_trabajo, "mejores_epochs.json")
with open(output_json_path, 'w') as f:
    json.dump(resultados, f, indent=4)

print(f"Archivo de salida guardado en: {output_json_path}")
