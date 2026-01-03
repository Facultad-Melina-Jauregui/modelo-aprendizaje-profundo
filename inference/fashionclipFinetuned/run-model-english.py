"""
Script para probar modelos fine-tuneados con descripciones en inglés.

Este script carga un modelo fine-tuneado, procesa imágenes de testing (removiendo
fondos si es necesario), y ejecuta múltiples pruebas de clasificación de texto-imagen
usando descripciones en inglés. Evalúa el rendimiento del modelo en diferentes tipos
de jeans (plain, ripped, wide leg, skinny, straight, palazzo, cargo, mom, flared)
y muestra métricas de clasificación para cada caso.
"""
import os
from PIL import Image
from rm_bg import remove_background
import numpy as np
from transformers import AutoProcessor, AutoModel
import torch
import torch.optim as optim
from fashionClipTestingText import test_text_clasification, find_similarities_matrix2

# cd database && python3 run-model.py > test.txt

# --- CONFIGURACIÓN ---
ORIGINAL_MODEL_NAME = "Marqo/marqo-fashionSigLIP"
MODEL_NAME_TO_FINETUNE = "Marqo/marqo-fashionSigLIP"
MODEL_NAME_TO_PUSH = "melijauregui/fashionSigLIP-roturas30"
FOLDER_IMAGES_TESTING = "images-testing"


BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-5
PATIENCE = 3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- TESTS ---
# print the info of the model
print("Model name:", MODEL_NAME_TO_PUSH)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME_TO_PUSH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=ORIGINAL_MODEL_NAME, trust_remote_code=True)
model.eval()
torch.manual_seed(42)

input_folder = FOLDER_IMAGES_TESTING
output_folder = "images-testing-nob"
if not os.path.exists(output_folder):
    remove_background(input_folder, output_folder)

image_paths = []
for root, _, files in os.walk(output_folder):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))


images = [Image.open(p).convert("RGB") for p in image_paths]

# --- test texto con descripcion de roturas vs liso
description = "plain jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=False, clasification_img="rotura")
print("\n")

description = "ripped jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="rotura")
print("\n")

# --- test texto con descripcion de wide leg
description = "wide leg jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="wide", yellow_flags=["palazzo"])
print("\n")

# --- test texto con descripcion de skinny
description = "skinny jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="skinny")
print("\n")

# --- test texto con descripcion de recto
description = "straight jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="recto",
                        yellow_flags=["mom"])
print("\n")

# --- test texto con descripcion de palazzo
description = "palazzo jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="palazzo", yellow_flags=["wide"])
print("\n")

# --- test texto con descripcion de cargo
description = "cargo jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="cargo")
print("\n")

# --- test texto con descripcion de mom
description = "mom jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="mom",
                        yellow_flags=["recto"])
print("\n")

# --- test texto con descripcion de flared
description = "flared jeans"
print("\nDescripcion:", description)
probabilities = find_similarities_matrix2(
    model, processor, description, image_paths, images)
test_text_clasification(probabilities=probabilities,
                        image_paths=image_paths, has=True, clasification_img="flared")
print("\n")
