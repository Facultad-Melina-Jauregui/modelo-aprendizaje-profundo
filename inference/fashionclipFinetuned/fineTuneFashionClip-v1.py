"""
Script de fine-tuning básico para modelos FashionCLIP/FashionSigLIP.

Este script entrena modelos de visión-lenguaje (CLIP/SigLIP) en datasets de moda
usando pérdida contrastiva. Carga imágenes desde URLs, procesa pares imagen-texto,
y actualiza los embeddings del modelo para mejorar la alineación entre imágenes y descripciones.
"""
from torch.nn.functional import cosine_similarity
from huggingface_hub import HfApi, HfFolder, Repository
from io import BytesIO
import os
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import create_repo, upload_file


# --- CONFIGURACIÓN ---
ORIGINAL_MODEL_NAME = "Marqo/marqo-fashionCLIP"
MODEL_NAME = "Marqo/marqo-fashionCLIP"
MODEL_NAME_TO_PUSH = "melijauregui/fashionclip-roturas4"
CSV_PATH = "datasets/roturas-vs-sin.csv"
BATCH_SIZE = 20
EPOCHS = 30
LR = 1e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --- DATASET PERSONALIZADO ---
class FashionDataset(Dataset):
    """
    Dataset personalizado que carga imágenes desde URLs y las asocia con descripciones de texto.
    """
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        response = requests.get(row["image"])
        image = Image.open(BytesIO(response.content))
        text = row["description"]
        return image, text


# --- FUNCIONES AUXILIARES ---
def collate_fn(batch):
    """
    Función de collate personalizada que agrupa imágenes y textos en listas.
    """
    images, texts = zip(*batch)
    return list(images), list(texts)


def contrastive_loss(image_embeds, text_embeds, margin=0.5):
    """
    Calcula la pérdida contrastiva entre embeddings de imagen y texto.

    Usa similitud coseno para medir la distancia positiva entre pares emparejados
    y aplica una función de pérdida con margen para maximizar la similitud de pares correctos.
    """
    positive = cosine_similarity(image_embeds, text_embeds)
    negative = 1 - positive
    loss = torch.mean(torch.relu(margin - positive + negative))
    return loss


def store_csv(username, dataset_name, csv_path):
    """
    Sube un archivo CSV a un repositorio de dataset en Hugging Face Hub.
    """
    repo_id = f"{username}/roturas"  # nombre del repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    upload_file(
        path_or_fileobj=csv_path,
        path_in_repo=dataset_name,  # cómo se llamará el archivo en el repo
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Dataset subido: https://huggingface.co/datasets/{repo_id}")


# --- CARGA DE DATOS Y MODELO ---
def fine_tune(csv_path, original_model_name, model_name, model_name_to_push, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """
    Entrena un modelo FashionCLIP/SigLIP en un dataset personalizado.

    Carga el modelo y processor, crea el dataloader, entrena el modelo usando
    pérdida contrastiva, y sube el modelo fine-tuneado al Hugging Face Hub.
    Soporta tanto modelos FashionCLIP como FashionSigLIP con procesamiento diferenciado.
    """
    df = pd.read_csv(csv_path)
    processor = AutoProcessor.from_pretrained(
        original_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True).to(DEVICE)

    dataset = FashionDataset(df, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # --- ENTRENAMIENTO ---
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, texts in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            if original_model_name.endswith("SigLIP"):
                inputs = processor(
                    images=images, text=texts, return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(DEVICE)
            else:
                inputs = processor(images=images, text=texts, return_tensors="pt",
                                   padding="max_length",
                                   truncation=True,
                                   max_length=77
                                   ).to(DEVICE)

            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            image_embeds = model.model.encode_image(pixel_values)
            text_embeds = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                normalize=True
            )

            # Normalizar (igual que get_*_features)
            image_embeds = image_embeds / \
                image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / \
                text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Calcular loss
            loss = contrastive_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    print(f'Pushing model to {model_name_to_push}')
    # --- GUARDAR MODELO ---
    model.push_to_hub(model_name_to_push)
    processor.push_to_hub(model_name_to_push)


fine_tune(csv_path="datasets/roturas-vs-sin.csv", original_model_name="Marqo/marqo-fashionSigLIP", model_name="melijauregui/fashionSigLIP-roturas",
        model_name_to_push="melijauregui/fashionSigLIP-roturas3")
store_csv("melijauregui", "roturas-vs-sin-roturas3.csv", "datasets/roturas-vs-sin.csv")


