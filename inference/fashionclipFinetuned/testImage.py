"""
Script para probar modelos fine-tuneados con múltiples conjuntos de imágenes (imagen-imagen).

Este script carga un modelo pre-entrenado, procesa imágenes de testing removiendo fondos,
y ejecuta pruebas para evaluar el rendimiento del modelo en diferentes categorías:
roturas/tipos de jeans, preferencias de estilo, y prendas generales. Cada prueba encuentra
la imagen más similar a una imagen query dentro de una galería de imágenes usando embeddings.
"""
import os
from PIL import Image
from rm_bg import remove_background
import numpy as np
from transformers import AutoProcessor, AutoModel
import torch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --- CONFIGURACIÓN ---
ORIGINAL_MODEL_NAME = "Marqo/marqo-fashionSigLIP"
MODEL_NAME_TO_PUSH = "melijauregui/cherrypick-best-sigLip"


# --- TESTS ---
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME_TO_PUSH, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=ORIGINAL_MODEL_NAME, trust_remote_code=True)
model.eval()
torch.manual_seed(42)


def find_similarities_image_to_images(model, processor, query_image_path, gallery_image_paths, gallery_images, green_flags=[], yellow_flags=[]):
    """
    Calcula similitudes entre una imagen query y una galería de imágenes usando embeddings.
    
    green_flags y yellow_flags son listas de nombres de archivo específicos.
    """
    query_image = Image.open(query_image_path).convert("RGB")

    with torch.no_grad():
        # Procesar query
        query_inputs = processor(
            images=query_image, return_tensors="pt").to(DEVICE)
        query_emb = model.model.encode_image(query_inputs["pixel_values"])
        query_emb = query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)

        # Procesar galería
        gallery_inputs = processor(
            images=gallery_images, return_tensors="pt").to(DEVICE)
        gallery_embs = model.model.encode_image(gallery_inputs["pixel_values"])
        gallery_embs = gallery_embs / gallery_embs.norm(p=2, dim=-1, keepdim=True)

    # Calcular similitudes
    similarities = torch.matmul(
        query_emb, gallery_embs.T).squeeze().cpu().numpy()

    image_names = [os.path.basename(path) for path in gallery_image_paths]
    query_name = os.path.basename(query_image_path)
    
    # Crear mapeo de nombre de archivo a índice
    name_to_index = {name: i for i, name in enumerate(image_names)}
    
    # Obtener índices y similitudes de green flags (manteniendo el orden de la lista)
    green_items = []
    green_names_set = set(green_flags)
    for filename in green_flags:
        if filename in name_to_index and filename != query_name:
            idx = name_to_index[filename]
            green_items.append((filename, similarities[idx]))
    
    # Obtener índices y similitudes de yellow flags (manteniendo el orden de la lista)
    yellow_items = []
    yellow_names_set = set(yellow_flags)
    for filename in yellow_flags:
        if filename in name_to_index and filename != query_name:
            idx = name_to_index[filename]
            yellow_items.append((filename, similarities[idx]))
    
    # Encontrar el máximo yellow flag (de la lista yellow_flags)
    max_yellow_flag = float('-inf')
    if yellow_items:
        max_yellow_flag = max(sim for _, sim in yellow_items)
    
    # Encontrar el máximo red flag (todo lo que NO está en green ni yellow, y no es la query)
    max_red_flag = float('-inf')
    best_red_name = None
    for i, name in enumerate(image_names):
        if name != query_name and name not in green_names_set and name not in yellow_names_set:
            if similarities[i] > max_red_flag:
                max_red_flag = similarities[i]
                best_red_name = name
    
    # Mostrar información de umbrales
    if best_red_name is not None:
        print(f"mayor red flag es {best_red_name} con {max_red_flag:.3f} de similitud")
    if yellow_items:
        max_yellow_name = max(yellow_items, key=lambda x: x[1])[0]
        print(f"mayor yellow flag es {max_yellow_name} con {max_yellow_flag:.3f} de similitud")
    
    # Evaluar y mostrar green_flags en el orden de la lista
    print("\ngreen_flags:")
    correct = 0
    correct_aprox = 0
    total = len(green_items)
    
    for filename, sim in green_items:
        if yellow_items and best_red_name:
            # Hay ambos umbrales
            success = sim > max_yellow_flag and sim > max_red_flag
            if success:
                correct += 1
            if sim > max_red_flag:
                symbol = '🟡'  # Pasa solo umbral de red
                correct_aprox += 1
            else:
                symbol = '❌'  # No pasa ni umbral de red
            if success:
                symbol = '✅'  # Pasa ambos umbrales
        elif best_red_name:
            # Solo hay red
            success = sim > max_red_flag
            if success:
                correct += 1
                symbol = '✅'
            else:
                symbol = '❌'
        else:
            symbol = '✅'
            correct += 1
        
        print(f"  {symbol} {filename}: {sim:.3f}")
    
    
    # Evaluar y mostrar yellow_flags en el orden de la lista
    if yellow_items:
        print("\nyellow_flags:")
        for filename, sim in yellow_items:
            if best_red_name:
                if sim > max_red_flag:
                    symbol = '✅'  # Pasa umbral red
                else:
                    symbol = '❌'  # No pasa umbral red
            else:
                symbol = '✅'
            
            print(f"  {symbol} {filename}: {sim:.3f}")
            
            
    # Calcular accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n\n🎯 Porcentaje de acierto: {accuracy:.2f}% ({correct}/{total})")
    if correct_aprox > correct:
        accuracy_aprox = (correct_aprox / total) * 100 if total > 0 else 0
        print(f"🎯 Porcentaje de acierto incluyendo 🟡: {accuracy_aprox:.2f}% ({correct_aprox}/{total})")
    
    min_yellow_flag = float('-inf')
    if yellow_items:
        min_yellow_flag = min(sim for _, sim in yellow_items)
        
    # Calcular Precision y Recall
    # Definir umbral para clasificar predicción positiva
    threshold_yellow = min_yellow_flag if yellow_items else float('-inf')
    threshold_red = max_red_flag if best_red_name else float('-inf')
    
    # True positives: todos los green flags (índices)
    true_positive = set()
    for filename, _ in green_items:
        if filename in name_to_index:
            true_positive.add(name_to_index[filename])
    
    true_positive_yellow = set()
    for filename, _ in yellow_items:
        if filename in name_to_index:
            true_positive_yellow.add(name_to_index[filename])
            
    print(f"true_positive: {true_positive}")
    print(f"true_positive_yellow: {true_positive_yellow}")
    
    # True negatives: red flags (no green, no yellow, no query)
    true_negative = set()
    for i, name in enumerate(image_names):
        if name != query_name and name not in green_names_set and name not in yellow_names_set:
            true_negative.add(i)
    
    # Predicciones binarias: si similitud > threshold -> pred positive
    predicted_positive = set()
    predicted_positive_yellow = set()
    for i, name in enumerate(image_names):
        if name != query_name and similarities[i] > threshold_red:
            predicted_positive.add(i)
        if name != query_name and similarities[i] > threshold_yellow :
            predicted_positive_yellow.add(i)
            
    print(f"predicted_positive: {predicted_positive}")
    print(f"predicted_positive_yellow: {predicted_positive_yellow}")
    
    # Precision = TP / (TP + FP)
    TP = len(predicted_positive.intersection(true_positive)) + len(predicted_positive.intersection(true_positive_yellow))
    FP = len(predicted_positive_yellow.difference(true_positive | true_positive_yellow))
    print(f"TP: {TP}, FP: {FP}")
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall = TP / (TP + FN)
    FN = len(true_positive.difference(predicted_positive)) + len(true_positive_yellow.difference(predicted_positive))
    print(f"FN: {FN}")
    print(f"no green flags que no son predicted: {len(true_positive.difference(predicted_positive))}")
    print(f"no yellow flags que no son predicted: {len(true_positive_yellow.difference(predicted_positive))}")
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    print(f"📊 Precision: {precision:.3f}")
    print(f"📊 Recall: {recall:.3f}")
    
    # MRR - buscar el primer ranking con imagen green flag
    sorted_indices = np.argsort(similarities)[::-1]
    mrr = 0
    first_rank = None
    green_indices_set = {name_to_index[filename] for filename, _ in green_items}
    
    for rank, idx in enumerate(sorted_indices, start=1):
        if idx in green_indices_set:
            mrr = 1 / rank
            first_rank = rank
            break
    
    print(f"📊 MRR (Mean Reciprocal Rank): {mrr:.3f}")
    if first_rank:
        print(f"📊 Primer green flag aparece en rank: {first_rank}")
    
    # --- RESUMEN FINAL ---
    print("\nRESUMEN FINAL:")
    accuracy_ratio = correct / total if total > 0 else 0
    print(f"{'❌' if accuracy_ratio < 0.6 else '✅'} Accuracy vs red/yellow flags (>= 0.6)")
    print(f"{'❌' if precision < 0.7 else '✅'} Precision (>= 0.7)")
    print(f"{'❌' if recall < 0.75 else '✅'} Recall (>= 0.75)")
    print(f"{'❌' if mrr < 0.3 else '✅'} MRR (>= 0.3)")
    if first_rank:
        print(f"{'❌' if first_rank > 3 else '✅'} Top-3 check")
    print("==============================")

    return similarities


def run_test(query_image_path, image_paths, images, green_flags=[], yellow_flags=[], test=True):
    """
    Ejecuta una prueba individual de búsqueda de imagen similar.

    Calcula similitudes entre la imagen query y todas las imágenes de la galería,
    y muestra los resultados ordenados por similitud con flags.
    
    Args:
        query_image_path: Ruta de la imagen query
        image_paths: Lista de rutas de imágenes en la galería
        images: Lista de imágenes PIL ya cargadas
        green_flags: Lista de nombres de archivo específicos que deberían aparecer primero (exactos)
        yellow_flags: Lista de nombres de archivo específicos similares pero no exactos
        test: Si es True, muestra los flags (por ahora siempre True)
    """
    query_basename = os.path.basename(query_image_path)
    print(f"Query imagen: {query_basename}")
    
    similarities = find_similarities_image_to_images(
        model, processor, query_image_path, image_paths, images, 
        green_flags=green_flags, yellow_flags=yellow_flags)
    
    print("\n")


def run_tests(input_folder, output_folder, run_func, test=True):
    """
    Procesa imágenes de una carpeta y ejecuta una función de pruebas.

    Remueve fondos de las imágenes, carga todas las imágenes procesadas,
    y ejecuta la función run_func con las imágenes cargadas.
    """
    remove_background(input_folder, output_folder)

    image_paths = []
    for root, _, files in os.walk(output_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    images = [Image.open(p).convert("RGB") for p in image_paths]

    run_func(image_paths, images, test)


def run_tests_queries(image_paths, images, test, query_candidates, get_queries_with_flags_func):
    
    if len(image_paths) == 0:
        print("No hay imágenes disponibles para testing")
        return
    
    # Encontrar las rutas completas de las queries
    query_paths_full = []
    for query_name in query_candidates:
        for path in image_paths:
            if os.path.basename(path) == query_name:
                query_paths_full.append(path)
                break
    
    for query_path in query_paths_full:
        flags_dict = get_queries_with_flags_func(os.path.basename(query_path))
        green_flags = flags_dict['green']
        yellow_flags = flags_dict['yellow']
        run_test(query_image_path=query_path, 
                image_paths=image_paths, 
                images=images,
                green_flags=green_flags,
                yellow_flags=yellow_flags,
                test=test)



def run_tests_roturas(image_paths, images, test):
    """
    Ejecuta pruebas para búsqueda de imágenes similares de roturas y tipos de jeans.

    Para cada imagen query de ejemplo (jean con roturas, liso, wide leg, etc.),
    busca las imágenes más similares en la galería con flags green/yellow/red.
    
    """
    
    query_candidates = [
        'wide-rotura-simetrica-negro1.png',
        'wide-rotura-simetrica1.png'
        'skinny-rotura-simetrica2.png',
        'recto-rotura-desmechada1.png',
        'cargo-vintage1.png',
        'flared1.png',
        'mom1.png',
        'palazzo2.png',
        'recto1.png',
        'skinny-oscuro.png',
        'skinny1.png',
        'wide1.png',
    ]
    run_tests_queries(image_paths, images, test, query_candidates, get_queries_with_flags_roturas)
    
    
def get_queries_with_flags_roturas(query_path):
    """
    Obtiene las queries con sus green y yellow flags
    """
    
    query_candidates = {
        'wide-rotura-simetrica-negro1.png': {'green': ['wide-rotura-simetrica-negro2.png', ], 'yellow': ['wide-rotura-simetrica2.png','recto-rotura-desmechada1.png','wide-rotura-simetrica1.png','wide-rotura-simetrica-desmechada1.png']},
        'wide-rotura-simetrica1.png': {'green': ['wide-rotura-simetrica-desmechada1.png', 'wide-rotura-simetrica2.png', 'recto-rotura4.png'], 'yellow': ['recto-rotura-simetrica1.png','recto-rotura-desmechada1.png', 'wide-rotura-simetrica-negro2.png', 'wide-rotura-simetrica-negro1.png']},
        'skinny-rotura-simetrica2.png': {'green': [ 'skinny-rotura-desmechada1.png', 'skinny-rotura-simetrica1.png', 'skinny-rotura-simetrica3.png', 'mom-rotura-desmechada2.png'], 'yellow': ['mom3.png', 'mom4.png']},
        'recto-rotura-desmechada1.png': {'green': ['wide-rotura-simetrica-desmechada1.png', 'recto-rotura-simetrica1.png', 'mom-rotura-desmechada2.png', 'recto-rotura4.png'], 'yellow': ['recto4.png', 'wide-rotura-simetrica1.png']},
        'cargo-vintage1.png': {'green': ['cargo1.png', 'cargo-vintage2.png', 'wide2.png'], 'yellow': ['cargo-azul1.png', 'cargo-grey1.png']},
        'flared1.png': {'green': ['flared2.png', 'flared3.png', 'flared4.png', 'flared5.png', 'flared6.png'], 'yellow': ['flared7.png']},
        'mom1.png': {'green': ['mom2.png', 'mom3.png', 'mom4.png', 'mom5.png', 'mom6.png'], 'yellow': ['mom-grey1.png', 'palazzo1.png']},
        'palazzo2.png': {'green': ['palazzo5.png', 'palazzo7.png'], 'yellow': ['wide2.png', 'wide1.png', 'palazzo3.png', 'palazzo4.png', 'palazzo6.png']},
        'recto1.png': {'green': ['recto2.png', 'recto3.png', 'recto4.png'], 'yellow': ['mom4.png']},
        'skinny-oscuro.png': {'green': ['skinny-oscuro2.png'], 'yellow': ['skinny1.png']},
        'skinny1.png': {'green': ['skinny-oscuro.png','skinny-oscuro2.png'], 'yellow': ['flared6.png', 'flared2.png', 'flared1.png']},
        'wide1.png': {'green': ['wide2.png', 'recto4.png', 'recto3.png'], 'yellow': ['palazzo7.png']},
    }
    
    return query_candidates[query_path]


def get_queries_with_flags_general(query_path):
    """
    Obtiene las queries con sus green y yellow flags
    """
    
    query_candidates = {
        'camisa-celeste-manga-larga.png': {'green': ['camisa-celeste-blanca-manga-larga.png',  ], 'yellow': ['camisa-rosa-manga-larga.png', 'camisa-celeste-manga-corta.png']},
        'campera-cuero-negra.png': {'green': ['campera-cuero-roja.png'], 'yellow': ['campera-jean-negra.png', 'campera-jean-azul.png']},
        'campera-jean-azul.png': {'green':['campera-jean-celeste.png'], 'yellow': ['campera-jean-negra.png']},
        'remera-celeste1.png': {'green':['remera-celeste2.png'], 'yellow': ['remera-celeste-blanca.png', 'remera-azul1.png']},
        'sweater-rosa.png': {'green':['sweater-rosa2.png','sweater-rosa3.png'], 'yellow': ['sweater-verde.png']},
        'vestido-estampado1.png': {'green':['vestido-estampado2.png', 'vestido-estampado3.png', 'vestido-rojo-estampado.png'], 'yellow': ['vestido-rojo-liso1.png','vestido-rojo-liso2.png', 'vestido-celeste-liso.png','vestido-verde-liso.png']},
        'vestido-rojo-liso1.png': {'green': ['vestido-rojo-liso2.png', 'vestido-rojo-estampado.png'], 'yellow': ['vestido-estampado1.png', 'vestido-estampado2.png']},
    }
    
    return query_candidates[query_path]



def run_tests_general(image_paths, images, test):
    """
    Ejecuta pruebas para búsqueda de imágenes similares de tipos de prendas generales.

    Para cada imagen query de ejemplo (remeras, sweaters, vestidos, camperas, camisas),
    busca las imágenes más similares en la galería con flags green/yellow/red.
    
    """
    
    query_candidates = [
        'camisa-celeste-manga-larga.png',
        'campera-cuero-negra.png',
        'campera-jean-azul.png',
        'remera-celeste1.png',
        'sweater-rosa.png',
        'vestido-estampado1.png',
        'vestido-rojo-liso1.png',
    ]
    run_tests_queries(image_paths, images, test, query_candidates, get_queries_with_flags_general)
    


if __name__ == "__main__":
    input_folder = "images-testing-roturas"
    output_folder = "images-testing-roturas-nobg"
    run_tests(input_folder, output_folder, run_tests_roturas)


    input_folder = "images_testing_general"
    output_folder = "images-testing-general-nobg"
    run_tests(input_folder, output_folder, run_tests_general)
