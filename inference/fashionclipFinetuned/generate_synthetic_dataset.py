"""
Script para generar datasets sintéticos mediante aumento de datos con modelos de lenguaje.

Este script utiliza modelos de generación de texto (Gemma) para crear descripciones
alternativas de prendas de ropa, aumentando el tamaño del dataset con variaciones
sintéticas de las descripciones originales. También incluye funciones para augmentation
de texto usando técnicas de NLP (sinónimos, intercambio de palabras, sustitución contextual).
"""
import pandas as pd
from transformers import pipeline
import torch
import nlpaug.augmenter.word as naw
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cpu",)

# pipe = pipeline("text-generation", model="gpt2", device="cpu")
# pipe = pipeline("text-generation", model="distilgpt2", device="cpu")
# pipe = pipeline("text-generation", model="facebook/bart-large", device="cpu")
# pipe = pipeline("text-generation",
#                model="flax-community/gpt-2-spanish", device="cpu")


def generar_respuesta(text, num_aug_per_sample=3):
    """
    Genera descripciones alternativas de una prenda usando un modelo de lenguaje.

    Utiliza el modelo Gemma para generar múltiples variaciones de una descripción
    en español, formateadas como opciones numeradas.
    """
    # prompt = (
    #    f"Generate {num_aug_per_sample} short alternative descriptions (1 paragraph each) in Spanish for the following clothing item. "
    #    "Do not repeat the prompt and do not include any other information. Just generate descriptions. "
    #    "Clothing: "
    # )

    prompt = (
        f"Generar {num_aug_per_sample} descripciones alternativas cortas (1 párrafo cada una) en español para la siguiente prenda de ropa."
        " Formatea la respuesta exactamente de esta manera:"
        "\n\n"
        "**Opción 1:** Texto de la descripción 1.\n"
        "**Opción 2:** Texto de la descripción 2.\n"
        "y continuar con las siguientes (sin texto adicional fuera de las opciones). Prenda: "
    )

    # output = pipe(prompt + text, max_length=256,
    #              num_return_sequences=num_aug_per_sample, truncation=True)
    # assistant_responses = [response['generated_text'].strip()
    #                       for response in output]
    messages = [
        {"role": "user", "content": f"{prompt} '{text}'"},
    ]
    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response


def parsear_respuestas(respuesta):
    """
    Parsea la respuesta del modelo extrayendo las descripciones de cada opción.

    Extrae las descripciones del formato "**Opción X:** descripción" y las limpia.
    """
    descripciones = []
    opciones = respuesta.split("**Opción ")
    for opcion in opciones[1:]:
        try:
            descripcion = opcion.split(":**")[1].strip()
            descripcion = limpiar_texto(descripcion)
            descripciones.append(descripcion)
        except IndexError:
            continue
    return descripciones


def limpiar_texto(texto):
    return texto.replace('"', '').replace("“", '').replace("”", '').strip()


synonym_aug = naw.SynonymAug(aug_p=0.4)
random_swap_aug = naw.RandomWordAug(action="swap")
contextual_aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute"
)

text_aug = [synonym_aug, random_swap_aug, contextual_aug]


def generate_synthetic_dataset2(df):
    """
    Genera un dataset sintético usando técnicas de augmentation de texto (NLPAug).

    Aplica augmentation de sinónimos, intercambio de palabras y sustitución contextual
    a cada descripción, generando múltiples variaciones por imagen.
    """
    synthetic_data = []
    for i, row in df.iterrows():
        print(f"Processing image {i+1}/{len(df)}: {row['image']}")
        synthetic_data.append(
            {"image": row["image"], "description": row["description"], "synthetic": False})
        for aug in text_aug:
            augmented_text = aug.augment(row["description"])
            print(f"Augmenting description {i+1}: {augmented_text}")
            synthetic_data.append(
                {"image": row["image"], "description": augmented_text, "synthetic": True})
        if i == 10:
            break
    synthetic_df = pd.DataFrame(synthetic_data)
    return synthetic_df


def generate_synthetic_dataset(df, num_aug_per_sample=3):
    """
    Genera un dataset sintético usando un modelo de lenguaje para crear descripciones alternativas.

    Para cada imagen, mantiene la descripción original y genera múltiples descripciones
    sintéticas usando el modelo Gemma. Marca las descripciones originales y sintéticas
    con la columna 'synthetic'.
    """
    synthetic_data = []
    for i, row in df.iterrows():
        # if i == 10:
        #    break
        print(f"Processing image {i+1}/{len(df)}: {row['image']}")
        synthetic_data.append(
            {"image": row["image"], "description": row["description"], "synthetic": False})
        try:
            texto_generado = generar_respuesta(
                row["description"], num_aug_per_sample)
            descripciones = parsear_respuestas(texto_generado)
        except Exception as e:
            print(f"Error procesando la imagen {row['image']}: {e}")
            continue

        for descripcion in descripciones:
            print(f"    - Description for image {i+1}: {descripcion}")
            synthetic_data.append(
                {"image": row["image"], "description": descripcion, "synthetic": True})
    synthetic_df = pd.DataFrame(synthetic_data)
    return synthetic_df


if __name__ == "__main__":
    # Ejemplo:
    # text = "jean de mujer wide leg sin rotura. Estampado blanco de estrellas. Podria ser palazzo o recto por su silueta. Tiro alto. Color negro destenido"
    # respuesta = generar_respuesta(text)
    # print(respuesta)
    df = pd.read_csv("datasets/con-sin-roturas-v3.csv")
    synthetic_df = generate_synthetic_dataset(df)
    synthetic_df.to_csv("datasets/cs-roturas-synthetic.csv",
                        index=False, quotechar='"')
