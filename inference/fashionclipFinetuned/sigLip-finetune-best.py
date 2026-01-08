"""
Script principal de fine-tuning para modelos FashionSigLIP con soporte para datasets unificados.

Este script proporciona funcionalidad completa para entrenar modelos FashionSigLIP en datasets
que combinan múltiples tareas (roturas y preferencias). Incluye soporte para data augmentation,
congelamiento selectivo de capas, early stopping, y evaluación con métricas completas
(recall@k, MRR, precisión). Soporta datasets unificados con columna 'task' para distinguir
entre diferentes tipos de clasificación.
"""
import math
import time
from sklearn.metrics import confusion_matrix
from torch.nn.functional import cosine_similarity
import os
from rembg import remove
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from transformers import AutoProcessor, AutoModel
import torch
import random
from torchvision import transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import multiprocessing
from nltk.corpus import stopwords
import re
from inference.fashionclipFinetuned.utils import *

# --- CONFIGURACIÓN ---
BATCH_SIZE = 30
EPOCHS = 30
LR = 1e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

NUM_WORKERS = min(4, os.cpu_count() or 1)
TEMPERATURE = 0.05

# print(f"Using {NUM_WORKERS} workers for data loading.")
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02))
])

# --- DATASET PERSONALIZADO ---


class FashionDataset(Dataset):
    """
    Dataset personalizado para modelos FashionSigLIP con soporte para datasets unificados.

    Carga imágenes desde rutas locales, aplica data augmentation opcional, y combina
    descripciones con tags. Soporta datasets con columna 'task' para balancear entre
    diferentes tareas (roturas/estilos) cuando se usa data augmentation.
    """
    def __init__(self, dataframe, processor, data_aug=False, n_augmented=5, estilo2idx=None):
        self.dataframe = dataframe
        self.processor = processor
        self.n_augmented = n_augmented
        self.data_aug = data_aug
        self.estilo2idx = estilo2idx or {}
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.RandomAffine(degrees=3, translate=(0.01, 0.01)),
        ])

    def __len__(self):
        if not self.data_aug:
            return len(self.dataframe)
        else:
            jeans = self.dataframe[self.dataframe["task"]
                                   == "roturas"].index.tolist()
            estilos = self.dataframe[self.dataframe["task"]
                                     == "estilos"].index.tolist()
            res_len = max(len(jeans), len(estilos)) * 2

            return res_len
            # return len(self.dataframe) * (self.n_augmented + 1)

    def __getitem__(self, idx):
        apply_aug = False
        if self.data_aug and isinstance(idx, tuple):
            #    base_idx = idx // (self.n_augmented + 1)
            #    apply_aug = idx % (self.n_augmented + 1) != 0
            #    idx = base_idx
            idx, apply_aug = idx

        row = self.dataframe.iloc[idx]

        filename = row["image"]
        # Normalizar rutas: convertir backslashes a forward slashes y hacer ruta relativa al directorio inference
        filename = filename.replace("\\", "/")
        if not os.path.isabs(filename):
            # Si es relativa, asumir que es relativa al directorio inference
            filename = os.path.join("inference", filename)

        try:
            image = Image.open(filename).convert("RGB")
            if apply_aug:
                image = self.augment(image)
        except Exception as e:
            print(f"⚠️ Error al cargar la imagen {filename}: {e}")
            raise e
            # return None

        text = row["description"]
        if row.get("tags", "") != "":
            if text != "" and not text.endswith("."):
                text += "."
            # text += f" Etiquetas: {row['tags']}"
            text += f" {row['tags']}"

        return image, text


def fine_tune(csv_path, original_model_name, model_name, model_name_to_push,
              batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, data_aug=False,
              freeze_func=freeze_layers, n_layers=4, loss_func=contrastive_loss):
    """
    Entrena un modelo FashionSigLIP en un dataset unificado con múltiples tareas.

    Divide el dataset en train/val, congela capas selectivamente, entrena con validación
    periódica, implementa early stopping basado en pérdida de validación, y sube el mejor
    modelo al Hugging Face Hub. Soporta datasets unificados que combinan roturas y preferencias.
    """
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    processor = AutoProcessor.from_pretrained(
        original_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True).to(DEVICE)

    freeze_func(model, n_layers=n_layers)

    # train_sampler = BalancedBatchSampler(train_df, batch_size=batch_size)

    # use_data_aug = data_aug or train_sampler.use_data_augmentation()

    train_dataset = FashionDataset(
        train_df, processor, data_aug=data_aug)

    val_dataset = FashionDataset(
        val_df, processor, data_aug=False)

    # train_loader = DataLoader(
    #    train_dataset,
    #    batch_sampler=train_sampler,
    #    collate_fn=collate_fn,
    #    num_workers=NUM_WORKERS
    # )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    params_to_optimize = []
    params_frozen = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
        else:
            params_frozen.append(param)

    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # # --- ENTRENAMIENTO ---
    patience = 3
    best_loss = float('inf')
    counter = 0
    model.train()
    best_model_state_dict = None

    for epoch in range(epochs):
        running_loss = 0.0
        for images, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
            loss = loss_func(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

        # evaluate(model, train_loader, processor, loss_func,
        #         desc=f"Epoch {epoch+1}/{epochs} [Train]")

        avg_val_loss = evaluate(
            model, val_loader, processor, loss_func, desc=f"Epoch {epoch+1}/{epochs} [Validation]"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            # Guardar el mejor modelo
            best_model_state_dict = model.state_dict()
            print(f"✨ Nuevo mejor modelo guardado en epoch {epoch+1}")
        else:
            counter += 1
            print(
                f"⚠️ No hay mejora en la pérdida. Patience: {counter}/{patience}")
            if counter >= patience:
                print("🛑 Early stopping activado.")
                break
        if avg_val_loss < 0.01:
            print("🛑 Early stopping activado por pérdida baja.")
            break

        # scheduler.step(avg_val_loss)

    # --- GUARDAR MODELO ---
    if best_model_state_dict:
        print(f'Pushing model to {model_name_to_push}')
        model.load_state_dict(best_model_state_dict)
        model.save_pretrained(model_name_to_push)
        processor.save_pretrained(model_name_to_push)
        model.push_to_hub(model_name_to_push)
        processor.push_to_hub(model_name_to_push)


def evaluate(model, loader, processor, loss_func, desc):
    """
    Evalúa el modelo calculando pérdida y métricas de rendimiento completas.

    Calcula pérdida promedio, recall@k para múltiples valores de k, MRR (Mean Reciprocal Rank),
    y precisión@5. Acumula todos los embeddings para calcular métricas globales sobre el conjunto completo.
    """
    model.eval()
    val_loss = 0.0
    all_image_embeds = []
    all_text_embeds = []
    all_texts = []
    with torch.no_grad():
        for images, texts in tqdm(loader, desc=desc):
            if not images:
                continue

            inputs = processor(
                images=images, text=texts, return_tensors="pt",
                padding=True,
                truncation=True,
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

            image_embeds = image_embeds / \
                image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / \
                text_embeds.norm(p=2, dim=-1, keepdim=True)

            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)
            all_texts.extend(texts)

            loss = loss_func(image_embeds, text_embeds)
            val_loss += loss.item()

    avg_loss = val_loss / len(loader)
    image_embeds = torch.cat(all_image_embeds)
    text_embeds = torch.cat(all_text_embeds)

    recalls = compute_recall_at_ks(image_embeds, text_embeds)
    mrr = compute_mrr(image_embeds, text_embeds)
    precision = compute_precision(image_embeds, text_embeds)
    print(f"🧪 Loss: {avg_loss:.4f}")
    for k, v in recalls.items():
        print(f"Recall 🔍 {k}: {v:.4f}")
    print(f"🔍 MRR: {mrr:.4f}")
    print(f"🔍 Precision@{5}: {precision:.4f}")
    return avg_loss


if __name__ == "__main__":
    multiprocessing.freeze_support()
    start_time = time.time()
    fine_tune(csv_path="inference/datasets/unificado/roturas-preferencias-v6.csv", original_model_name="Marqo/marqo-fashionSigLIP", model_name="Marqo/marqo-fashionSigLIP",
              model_name_to_push="melijauregui/cherrypick-best-sigLip", data_aug=False,
              loss_func=contrastive_loss_InfoNCE, batch_size=16, epochs=32, lr=2e-5,
              n_layers=2)
    elapsed_time = time.time() - start_time
    print(f"⏱ Tiempo total de ejecución: {format_time(elapsed_time)}")
