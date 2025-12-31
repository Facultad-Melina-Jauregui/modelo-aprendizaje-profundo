"""
Script para detectar imágenes duplicadas en datasets CSV.

Este script proporciona funciones para calcular hashes de imágenes y detectar
filas duplicadas en un CSV basándose en las URLs de las imágenes.
"""
import pandas as pd
import hashlib
from PIL import Image
from pathlib import Path


def hash_image(image_path):
    """
    Calcula el hash MD5 de una imagen para identificar duplicados por contenido.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception as e:
        print(f"Error leyendo {image_path}: {e}")
        return None


def detectar_duplicadas(csv_path):
    """
    Detecta y muestra filas duplicadas en un CSV basándose en la columna 'image'.
    """
    df = pd.read_csv(csv_path)

    duplicados = df[df.duplicated(subset=["image"], keep=False)]

    if duplicados.empty:
        print("No se encontraron imágenes duplicadas por URL.")
    else:
        print(
            f"Se encontraron {duplicados.shape[0]} filas con URLs duplicadas:")
        for index, row in duplicados.iterrows():
            print(f"Fila {index}: {row['image']}, {row['description']}")


if __name__ == "__main__":
    detectar_duplicadas("datasets/con-sin-roturas-v4.csv")
