import subprocess
import os
import shutil

def recortar_video(input_path, output_path, start_time=60, duration=400):
    """
    Recorta un vídeo entre start_time y start_time + duration usando FFmpeg.y


    Parámetros:
    - input_path: Ruta del vídeo de entrada.
    - output_path: Ruta donde se guardará el vídeo recortado.
    - start_time: Segundo de inicio del recorte (por defecto 20).
    - duration: Duración del recorte en segundos (por defecto 20).
    """

    # Verifica que FFmpeg esté disponible
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg no está instalado o no está en el PATH del sistema.")

    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Vídeo recortado guardado en: {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error al recortar el vídeo:", e)

# Ejemplo de uso:
if __name__ == "__main__":
    input_video = "/home/gmanty/code/AnemoNAS/14-12-23/todo/USCL2-074558-075058.mp4"
    output_video = "/home/gmanty/code/AnemoNAS/14-12-23/todo/USCL2-074558-075058_rec.mp4"
    recortar_video(input_video, output_video)
