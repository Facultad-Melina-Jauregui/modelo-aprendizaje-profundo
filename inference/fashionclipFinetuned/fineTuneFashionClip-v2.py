"""
Script de fine-tuning mejorado para modelos FashionCLIP/FashionSigLIP con validación y early stopping.

Este script extiende la versión v1 agregando división train/validation, evaluación en validación,
y early stopping para evitar sobreentrenamiento. Guarda el mejor modelo según la pérdida de validación.
"""
from sklearn.model_selection import train_test_split
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu")

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
        image = Image.open(BytesIO(response.content)).convert("RGB")
        text = row["description"]
        return image, text


# --- FUNCIONES AUXILIARES ---
def collate_fn(batch):
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


@torch.no_grad()
def evaluate(model, val_loader, processor):
    """
    Evalúa el modelo en un conjunto de validación calculando la pérdida promedio.

    Desactiva el cálculo de gradientes y calcula la pérdida contrastiva en el conjunto de validación.
    """
    model.eval()
    total_loss = 0
    for images, texts in val_loader:
        inputs = processor(images=images, text=texts, return_tensors="pt",
                           padding=True, truncation=True).to(DEVICE)

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        image_embeds = model.model.encode_image(pixel_values)
        text_embeds = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=True
        )

        image_embeds = image_embeds / \
            image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        loss = contrastive_loss(image_embeds, text_embeds)
        total_loss += loss.item()
    return total_loss / len(val_loader)


# --- ENTRENAMIENTO ---
def fine_tune(csv_path, original_model_name, model_name, model_name_to_push, batch_size, epochs, lr, patience):
    """
    Entrena un modelo FashionCLIP/SigLIP con división train/val y early stopping.

    Divide el dataset en entrenamiento (80%) y validación (20%), entrena el modelo,
    implementa early stopping basado en pérdida de validación, guarda el mejor modelo,
    y lo sube al Hugging Face Hub al finalizar.
    """
    df = pd.read_csv(csv_path)

    # División entrenamiento / validación
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True)

    processor = AutoProcessor.from_pretrained(
        original_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True).to(DEVICE)

    train_dataset = FashionDataset(train_df, processor)
    val_dataset = FashionDataset(val_df, processor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = processor(images=images, text=texts, return_tensors="pt",
                               padding=True, truncation=True).to(DEVICE)

            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            image_embeds = model.model.encode_image(pixel_values)
            text_embeds = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                normalize=True
            )

            image_embeds = image_embeds / \
                image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / \
                text_embeds.norm(p=2, dim=-1, keepdim=True)

            loss = contrastive_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, processor)

        print(
            f"✅ Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping activado.")
                break

    # Restaurar el mejor modelo
    model.load_state_dict(torch.load("best_model.pt"))

    print(f'📤 Pushing modelo a {model_name_to_push}')
    model.push_to_hub(model_name_to_push)
    processor.push_to_hub(model_name_to_push)

# fine_tune(csv_path="datasets/con-sin-roturas.csv", original_model_name=model, model_name=model,
#           model_name_to_push="melijauregui/fashionSigLIP-roturas3", batch_size=32, epochs=15)
