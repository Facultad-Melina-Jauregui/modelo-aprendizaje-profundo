"""
Script para procesar imágenes de un dataset removiendo el fondo.

Este script lee un CSV con URLs de imágenes, descarga cada imagen, le remueve el fondo
usando la biblioteca rembg, y guarda las imágenes procesadas localmente. Actualiza el CSV
con las rutas locales a las imágenes procesadas. Organiza las imágenes en directorios
según su tipo (preferencias o roturas) basándose en los tags.
"""
import os
import pandas as pd
from PIL import Image
from rembg import remove
from io import BytesIO
import requests
from tqdm import tqdm

INPUT_CSV = "datasets/unificado/roturas-preferencias-v6.csv"
OUTPUT_CSV = "datasets/unificado/roturas-preferencias-v6.csv"


df = pd.read_csv(INPUT_CSV)
processed_entries = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row["image"]
    description = row["description"]
    tags = row["tags"]
    task = row["task"]

    try:
        # Nombre de archivo base
        basename = os.path.basename(url).split("?")[0]
        # directorio
        dir_name = os.path.dirname(url)
        rm = False
        if not url.endswith('nobg.png'):
            if url.endswith('.jpg'):
                basename = basename.replace(
                    ".jpg", "_nobg.png")
            else:
                basename = basename.replace(".png", "_nobg.png")
            rm = True

        # si no existe el dir_name
        if not os.path.exists(dir_name):
            if tags.startswith("Estilos:"):
                dir_name = "processed_preferences_images"
            else:
                dir_name = "processed_images_roturas"
            os.makedirs(dir_name, exist_ok=True)
        output_filename = os.path.join(dir_name, basename)
        if url.endswith('_nobg_nobg.png'):
            output_filename = output_filename.replace(
                "_nobg_nobg.png", "_nobg.png")
            os.rename(os.path.join(dir_name, basename), output_filename)

        if rm and not os.path.exists(output_filename):
            # Descargar la imagen
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Eliminar fondo
            image_nobg = remove(image)
            image_nobg.save(output_filename)

        processed_entries.append({
            "image": output_filename,
            "description": description,
            "tags": tags,
            "task": task
        })

    except Exception as e:
        print(f"⚠️ Error con imagen {url}: {e}")

# Guardar nuevo CSV con rutas locales
processed_df = pd.DataFrame(processed_entries)
processed_df.to_csv(OUTPUT_CSV, index=False)

print(
    f"✅ Proceso terminado. Imágenes guardadas y CSV generado en '{OUTPUT_CSV}'.")
