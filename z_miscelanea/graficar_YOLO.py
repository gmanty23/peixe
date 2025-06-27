import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparative_metrics(file1, file2):
    # Cargar los datos
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Filtrar últimas 35 epochs
    df1_last = df1.tail(35)
    df2_last = df2.tail(35)
    
    # Métricas a comparar
    metrics = [
        ('metrics/mAP50(B)', 'mAP@50'),
        ('metrics/mAP50-95(B)', 'mAP@50-95')
    ]
    
    # Carpeta de salida
    output_dir = os.path.dirname(file1)
    
    # Estilo formal
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150
    })
    
    for metric_col, metric_name in metrics:
        plt.figure(figsize=(8, 5))
        
        plt.plot(df1_last['epoch'], df1_last[metric_col], marker='o', linestyle='-', label='640')
        plt.plot(df2_last['epoch'], df2_last[metric_col], marker='s', linestyle='-', label='1024')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Evolución de {metric_name} en las últimas 35 epochs')
        plt.grid(True, linestyle=':', linewidth=0.7)
        plt.legend()
        
        # Guardar
        filename = f"comparativa_{metric_col.replace('/', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Gráfico guardado en: {save_path}")

    

# Ejemplo de uso:
# plot_comparative_metrics("/ruta/al/primer/archivo.csv", "/ruta/al/segundo/archivo.csv")
if __name__ == "__main__":
    # Reemplaza con las rutas de tus archivos CSV
    file1 = '/home/gmanty/code/peixe/runs/detect/results_combined_640.csv'
    file2 = '/home/gmanty/code/peixe/runs/detect/results_combined_1024.csv'

    plot_comparative_metrics(file1, file2)
