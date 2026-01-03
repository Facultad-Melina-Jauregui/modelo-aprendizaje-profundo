"""
Script para remover fondos de imágenes en una carpeta usando rembg.

Este script procesa todas las imágenes de una carpeta, les remueve el fondo
usando la biblioteca rembg, y guarda las imágenes procesadas en formato PNG
en una carpeta de salida. Omite imágenes que ya han sido procesadas.
"""
import time
import os
from rembg import remove
from PIL import Image
import io


def process(input_folder, output_folder, img_name):
    """
    Procesa una imagen individual removiendo su fondo y guardándola en formato PNG.

    Si la imagen de salida ya existe, omite el procesamiento.
    """

    input_path = os.path.join(input_folder, img_name)
    output_path = os.path.join(output_folder, os.path.splitext(img_name)[
        0] + ".png")  # siempre PNG
    if os.path.exists(output_path):
        return

    print(f"Procesando {img_name}")

    with open(input_path, 'rb') as f:
        input_data = f.read()
        output_data = remove(input_data)

    img = Image.open(io.BytesIO(output_data))
    img.save(output_path)


def remove_background(input_folder, output_folder):
    """
    Procesa todas las imágenes de una carpeta removiendo sus fondos.

    Filtra archivos de imagen válidos (jpg, jpeg, png), procesa cada una
    usando la función process, y maneja errores individualmente sin interrumpir el proceso.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Filtrar archivos de imagen
    valid_extensions = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(
        input_folder) if f.lower().endswith(valid_extensions)]

    total_start = time.time()

    for img_name in images:
        try:
            process(input_folder, output_folder, img_name)

        except Exception as e:
            print(f"Error procesando {img_name}: {e}")

    total_elapsed = time.time() - total_start
    # print(f"\nFinalizado. Tiempo total: {total_elapsed:.2f} segundos.")


# remove_background(input_folder, output_folder)
