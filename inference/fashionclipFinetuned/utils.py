"""
Módulo de utilidades para fine-tuning y evaluación de modelos FashionSigLIP.

Este módulo proporciona funciones auxiliares y clases para:
- Samplers balanceados para datasets unificados
- Funciones de pérdida contrastiva (margen, InfoNCE, InfoNCE con hard negatives)
- Congelamiento selectivo de capas del modelo
- Métricas de evaluación (recall@k, MRR, precisión)
- Evaluación por clases y matrices de confusión
- Utilidades de formato y procesamiento
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
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoProcessor, AutoModel
import torch
import random
from torchvision import transforms
import torch.optim as optim
# pip install nlpaug transformers sentencepiece
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import multiprocessing
# import nltk
from nltk.corpus import stopwords
import re
# import spacy
# nlp = spacy.load("en_core_web_sm")  # python -m spacy download en_core_web_sm

# nltk.download('stopwords')
# stop_words_en = set(stopwords.words('english'))
# stop_words_es = set(stopwords.words('spanish'))

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


# def clean_text(text):
#    tokens = re.findall(r'\b\w+\b', text.lower())
#    filtered = [word for word in tokens if word not in stop_words_en]
#    return " ".join(filtered)

# def clean_text2(text):
#    doc = nlp(text.lower())
#    lemmatized = [
#        token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
#    return " ".join(lemmatized)


# --- DATASET PERSONALIZADO ---
class BalancedBatchSampler(Sampler):
    """
    Sampler que balancea batches entre diferentes tareas (roturas y estilos).

    Genera batches que contienen muestras balanceadas de ambas tareas,
    aplicando data augmentation a la categoría menor cuando hay desbalance.
    Mezcla muestras normales y augmentadas en los batches.
    """
    def __init__(self, dataframe, batch_size):
        self.batch_size = batch_size

        # Reindexar dataframe para que sea de 0..n-1
        df = dataframe.reset_index(drop=True)

        # Guardar índices por categoría
        self.indices_jeans = df[df["task"] == "roturas"].index.tolist()
        self.indices_estilos = df[df["task"] == "estilos"].index.tolist()

        assert self.indices_jeans and self.indices_estilos, "Faltan datos de alguna categoría"

    def use_data_augmentation(self):
        return len(self.indices_jeans) != len(self.indices_estilos)

    def __iter__(self):
        jeans = self.indices_jeans.copy()
        estilos = self.indices_estilos.copy()

        # Data aug a la categoría más chica para empatar tamaño
        if len(jeans) > len(estilos):
            diff = len(jeans) - len(estilos)
            if diff > len(estilos):
                augment_indices = random.choices(estilos, k=diff)
            else:
                augment_indices = random.sample(estilos, diff)

            estilos += [(idx, True) for idx in augment_indices]
        elif len(estilos) > len(jeans):
            diff = len(estilos) - len(jeans)
            if diff > len(estilos):
                augment_indices = random.choices(estilos, k=diff)
            else:
                augment_indices = random.sample(estilos, diff)
            jeans += [(idx, True) for idx in augment_indices]

        # Mezclar
        jeans = self.shuffle_mixed(jeans)
        estilos = self.shuffle_mixed(estilos)

        combined = [val for pair in zip(jeans, estilos) for val in pair]

        # Entregar en batches
        for i in range(0, len(combined), self.batch_size):
            yield combined[i:i+self.batch_size]

    def shuffle_mixed(self, lst):
        normal = [x for x in lst if not (
            isinstance(x, tuple) and x[1] is True)]
        augmented = [x for x in lst if isinstance(x, tuple) and x[1] is True]
        random.shuffle(normal)
        random.shuffle(augmented)
        return normal + augmented

    def __len__(self):
        jeans = len(self.indices_jeans)
        estilos = len(self.indices_estilos)
        max_len = max(jeans, estilos)

        total_items = max_len * 2
        res = math.ceil(total_items / self.batch_size)

        return res

# --- FUNCIONES AUXILIARES ---


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return [], []
    images, texts = zip(*batch)
    return list(images), list(texts)


def freeze_layers(model, n_layers):
    """
    Congela la mayoría de las capas del modelo y descongela selectivamente para fine-tuning.

    Congela todas las capas, luego descongela las últimas n_layers de visión y texto,
    más las proyecciones finales (attn_pool y text_projection) para permitir fine-tuning eficiente.
    """
    # Descongelar todas las capas
    for param in model.parameters():
        param.requires_grad = False

    # --- VISIÓN ---
    # Descongelar las últimas capas de visión
    if n_layers != 0:
        visual_blocks = model.model.visual.trunk.blocks
        for block in visual_blocks[-n_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    # Descongelar la attn_pool
    for param in model.model.visual.trunk.attn_pool.parameters():
        param.requires_grad = True

    # --- TEXTO ---
    # Descongelar las últimas capas de texto
    if n_layers != 0:
        text_layers = model.model.text.transformer.resblocks
        for layer in text_layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Descongelar la proyección de texto
    for param in model.model.text.text_projection.parameters():
        param.requires_grad = True


def contrastive_loss(image_embeds, text_embeds, margin=0.5):
    """
    Calcula pérdida contrastiva usando similitud coseno con margen.

    Usa una función de pérdida con margen para maximizar la similitud de pares correctos
    y minimizar la de pares incorrectos.
    """
    positive = cosine_similarity(image_embeds, text_embeds)

    negative = 1 - positive
    loss = torch.mean(torch.relu(margin - positive + negative))
    return loss


def contrastive_loss_InfoNCE_hard_negatives(text_embeds, image_embeds, temperature=TEMPERATURE):
    """
    Calcula pérdida InfoNCE con hard negatives adicionales.

    Combina pérdidas image-to-text, text-to-image y hard negatives para
    mejorar el aprendizaje con ejemplos difíciles.
    """
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

    logits = torch.matmul(text_embeds, image_embeds.T) / temperature
    labels = torch.arange(len(text_embeds)).to(logits.device)

    # Seleccionar hard negatives
    hard_negatives = logits.topk(k=1, dim=1, largest=True).indices.squeeze()

    loss_hard_negatives = F.cross_entropy(logits, hard_negatives)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i + loss_hard_negatives) / 3


def contrastive_loss_InfoNCE(text_embeds, image_embeds, temperature=TEMPERATURE):
    """
    Calcula pérdida InfoNCE (Info Noise Contrastive Estimation) estándar.

    Usa cross-entropy para maximizar la similitud entre pares correctos y minimizar
    la de pares incorrectos, promediando las pérdidas image-to-text y text-to-image.
    """
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

    logits = torch.matmul(text_embeds, image_embeds.T) / temperature
    labels = torch.arange(len(text_embeds)).to(logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


# def compute_recall_at_k(image_embeds, text_embeds, k=5):
#     sims = image_embeds @ text_embeds.T
#     targets = torch.arange(len(image_embeds), device=sims.device)
#     _, topk = sims.topk(k, dim=1)
#     correct = topk.eq(targets.unsqueeze(1)).any(dim=1).float()
#     return correct.mean().item()

def compute_recall_at_ks(image_embeds, text_embeds, ks=[1, 5, 10]):
    """
    Calcula recall@k para múltiples valores de k.

    Mide la proporción de imágenes cuya descripción correcta está en los top-k
    resultados más similares.
    """
    sims = image_embeds @ text_embeds.T
    targets = torch.arange(len(image_embeds), device=sims.device)
    recalls = {}
    for k in ks:
        _, topk = sims.topk(k, dim=1)
        correct = topk.eq(targets.unsqueeze(1)).any(
            dim=1).float().mean().item()
        recalls[f"recall@{k}"] = correct
    return recalls


def compute_mrr(image_embeds, text_embeds):
    """
    Calcula MRR (Mean Reciprocal Rank) entre embeddings de imagen y texto.

    Mide el ranking promedio de las descripciones correctas, donde el rank
    es la posición de la descripción correcta en la lista ordenada por similitud.
    """
    sims = image_embeds @ text_embeds.T
    targets = torch.arange(len(image_embeds), device=sims.device)
    sorted_indices = sims.argsort(dim=1, descending=True)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1
    reciprocal_ranks = 1.0 / ranks.float()
    return reciprocal_ranks.mean().item()


def compute_precision(image_embeds, text_embeds, k=5):
    """
    Calcula precisión@k entre embeddings de imagen y texto.

    Mide la proporción de imágenes cuya descripción correcta está en los top-k
    resultados más similares (similar a recall@k pero con normalización adicional).
    """
    image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

    similarity = image_embeds @ text_embeds.T
    topk = similarity.topk(k, dim=-1).indices
    matches = torch.arange(len(image_embeds)).unsqueeze(1).to(topk.device)
    precision = (topk == matches).any(dim=1).float().mean().item()
    return precision


def evaluate_by_class(image_embeds, text_embeds, df_val, ks=[1, 5, 10]):
    """
    Evalúa métricas de recall@k por clase (por ejemplo, con/sin roturas).

    Divide los embeddings por clase según la columna 'has_rips' del dataframe
    y calcula recall@k para cada clase por separado.
    """
    # df_val debe tener columnas 'cut_type' y 'has_rips'
    results = {}
    labels = df_val["has_rips"].values  # 0 o 1
    for cls in [0, 1]:
        idxs = (labels == cls).nonzero()[0]
        sub_img_emb = image_embeds[idxs]
        sub_txt_emb = text_embeds[idxs]
        recalls = compute_recall_at_ks(sub_img_emb, sub_txt_emb, ks)
        results[f"has_rips={cls}"] = recalls
    # idem podrías hacer por cut_type iterando sus valores únicos
    return results


def confusion_plain_vs_ripped(image_embeds, df_val, model, processor):
    """
    Calcula matriz de confusión para clasificación plain vs ripped jeans.

    Construye queries de texto para "plain jeans" y "ripped jeans", encuentra
    las imágenes más similares a cada query, y genera una matriz de confusión
    comparando con las etiquetas reales.
    """
    # construimos dos queries
    queries = ["plain jeans", "ripped jeans"]
    q_embeds = []
    for q in queries:
        inp = processor(text=[q], return_tensors="pt",
                        padding=True, truncation=True).to(DEVICE)
        q_emb = model.get_text_features(**inp, normalize=True)
        q_embeds.append(F.normalize(q_emb, dim=-1)[0])

    # para cada query obtenemos top-1
    sims = torch.stack(q_embeds) @ image_embeds.T  # (2, N_images)
    # índices top1 para cada query
    preds = sims.argmax(dim=1).cpu().numpy()

    # extraemos ground‐truth has_rips
    truths = [
        0,  # plain jeans -> esperamos has_rips=0
        1,  # ripped jeans -> esperamos has_rips=1
    ]
    pred_labels = df_val.iloc[preds]["has_rips"].values
    # Matriz de confusión
    cm = confusion_matrix(truths, pred_labels, labels=[0, 1])
    print("Confusion matrix (plain=0, ripped=1):\n", cm)


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"
