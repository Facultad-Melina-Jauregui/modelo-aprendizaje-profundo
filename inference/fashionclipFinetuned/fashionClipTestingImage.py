"""
Script para probar modelos de búsqueda de imágenes similares usando embeddings.

Este script carga un modelo fine-tuneado de FashionSigLIP/FashionCLIP y encuentra
la imagen más similar a una imagen query dentro de una galería de imágenes.
Utiliza embeddings normalizados y similitud coseno para comparar imágenes.
"""
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import numpy as np
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu")


def find_most_similar_image(query_image_path, pretrained_model_name, gallery_paths, model_name):
    """
    Encuentra la imagen más similar a una imagen query en una galería usando embeddings.

    Carga el modelo fine-tuneado y el processor base, genera embeddings normalizados
    para la imagen query y todas las imágenes de la galería, calcula similitudes usando
    producto punto (similitud coseno) y muestra los resultados ordenados por similitud.
    """
    folder_path = "images-testing"
    # Configurar dispositivo

    # Cargar modelo y processor
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, low_cpu_mem_usage=False).to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name, trust_remote_code=True)

    query_image = Image.open(
        f'{folder_path}/{query_image_path}').convert("RGB")

    gallery_images = [Image.open(
        f'{folder_path}/{p}').convert("RGB") for p in gallery_paths]

    with torch.no_grad():
        # Procesar query
        query_inputs = processor(
            images=query_image, return_tensors="pt").to(DEVICE)
        query_emb = model.model.encode_image(query_inputs["pixel_values"])
        query_emb = query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)

        gallery_inputs = processor(
            images=gallery_images, return_tensors="pt").to(DEVICE)
        gallery_embs = model.model.encode_image(gallery_inputs["pixel_values"])
        gallery_embs = gallery_embs / \
            gallery_embs.norm(p=2, dim=-1, keepdim=True)

    similarities = torch.matmul(
        query_emb, gallery_embs.T).squeeze().cpu().numpy()

    sorted_indices = np.argsort(similarities)[::-1]
    for i in sorted_indices:
        print(f"Similitud con '{gallery_paths[i]}': {similarities[i]}")

    print(
        f"📸 Imagen más similar para {query_image_path}: {gallery_paths[np.argmax(similarities)]}\n")


pretrained_model_name_or_path = "Marqo/marqo-fashionSigLIP"
model_name = "melijauregui/fashionSigLIP-roturas8"
query_image_path = "roturas-negro2.jpg"
gallery_paths = [
    "rotura1.png",
    "rotura2.png",
    "rotura3.png",
    "sin-rotura.png",
    "roturas-negro1.jpg",
    "skinny-rotura.png"
]
find_most_similar_image(query_image_path, pretrained_model_name_or_path, gallery_paths,
                        model_name=model_name)

query_image_path = "rotura1.png"
gallery_paths = [
    "rotura2.png",
    "rotura3.png",
    "sin-rotura.png",
    "roturas-negro1.jpg",
    "roturas-negro2.jpg",
    "skinny-rotura.png"
]
find_most_similar_image(query_image_path, pretrained_model_name_or_path, gallery_paths,
                        model_name=model_name)

""" 
pretrained_model_name_or_path = "Marqo/marqo-fashionSigLIP"
model_name = "melijauregui/fashionSigLIP-roturas"

Similitud con 'roturas-negro1.jpg': 0.9825956225395203
Similitud con 'rotura1.png': 0.9593182802200317
Similitud con 'rotura2.png': 0.9545838832855225
Similitud con 'rotura3.png': 0.9480644464492798
Similitud con 'sin-rotura.png': 0.8960565328598022
Similitud con 'skinny-rotura.png': 0.8698640465736389
📸 Imagen más similar para :roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.9768972396850586
Similitud con 'rotura2.png': 0.9755618572235107
Similitud con 'roturas-negro1.jpg': 0.9666265249252319
Similitud con 'roturas-negro2.jpg': 0.9593180418014526
Similitud con 'skinny-rotura.png': 0.903894305229187
Similitud con 'sin-rotura.png': 0.8768881559371948
📸 Imagen más similar para :rotura1.png: rotura3.png

------------------------

pretrained_model_name_or_path = "Marqo/marqo-fashionSigLIP"
model_name = "melijauregui/fashionSigLIP-roturas2"

Similitud con 'roturas-negro1.jpg': 0.9772899746894836
Similitud con 'rotura2.png': 0.9400231838226318
Similitud con 'rotura3.png': 0.9360625743865967
Similitud con 'rotura1.png': 0.9252451658248901
Similitud con 'sin-rotura.png': 0.9028607606887817
Similitud con 'skinny-rotura.png': 0.8645125031471252
📸 Imagen más similar para :roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.9816862344741821
Similitud con 'rotura2.png': 0.9697244167327881
Similitud con 'sin-rotura.png': 0.9368839263916016
Similitud con 'roturas-negro1.jpg': 0.9331836104393005
Similitud con 'skinny-rotura.png': 0.9270892143249512
Similitud con 'roturas-negro2.jpg': 0.9252451658248901
📸 Imagen más similar para :rotura1.png: rotura3.png

--------------------------

pretrained_model_name_or_path = "Marqo/marqo-fashionSigLIP"
model_name = "melijauregui/fashionSigLIP-roturas3"

Similitud con 'roturas-negro1.jpg': 0.9869186878204346
Similitud con 'rotura2.png': 0.9609938263893127
Similitud con 'rotura3.png': 0.9580990076065063
Similitud con 'rotura1.png': 0.9506805539131165
Similitud con 'sin-rotura.png': 0.917148232460022
Similitud con 'skinny-rotura.png': 0.8671223521232605
📸 Imagen más similar para :roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.9871813654899597
Similitud con 'rotura2.png': 0.97878098487854
Similitud con 'roturas-negro1.jpg': 0.9603569507598877
Similitud con 'roturas-negro2.jpg': 0.9506804943084717
Similitud con 'sin-rotura.png': 0.925117552280426
Similitud con 'skinny-rotura.png': 0.9138990640640259
📸 Imagen más similar para :rotura1.png: rotura3.png

-------------------------

pretrained_model_name_or_path = "Marqo/marqo-fashionCLIP"
model_name = "melijauregui/fashionclip-roturas3"

Similitud con 'roturas-negro1.jpg': 0.9827072620391846
Similitud con 'rotura2.png': 0.8716366291046143
Similitud con 'rotura3.png': 0.8712751269340515
Similitud con 'rotura1.png': 0.866313099861145
Similitud con 'sin-rotura.png': 0.8477672934532166
Similitud con 'skinny-rotura.png': 0.8231112957000732
📸 Imagen más similar para :roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.986778974533081
Similitud con 'rotura2.png': 0.9802199602127075
Similitud con 'sin-rotura.png': 0.9763138890266418
Similitud con 'skinny-rotura.png': 0.9402551651000977
Similitud con 'roturas-negro2.jpg': 0.8663132190704346
Similitud con 'roturas-negro1.jpg': 0.8636921644210815
📸 Imagen más similar para :rotura1.png: rotura3.png

---------------------------------------------

melijauregui/fashionSigLIP-roturas4: congelar capas y optimizar solo la última.

Similitud con 'roturas-negro1.jpg': 0.9166030883789062
Similitud con 'rotura2.png': 0.8889819979667664
Similitud con 'rotura3.png': 0.8875263929367065
Similitud con 'skinny-rotura.png': 0.8473618030548096
Similitud con 'rotura1.png': 0.8370959758758545
Similitud con 'sin-rotura.png': 0.6315427422523499
📸 Imagen más similar para :roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.916398286819458
Similitud con 'rotura2.png': 0.8704217076301575
Similitud con 'roturas-negro1.jpg': 0.8607035875320435
Similitud con 'skinny-rotura.png': 0.8460321426391602
Similitud con 'roturas-negro2.jpg': 0.8370959162712097
Similitud con 'sin-rotura.png': 0.5840163230895996
📸 Imagen más similar para :rotura1.png: rotura3.png

------------------------------------

melijauregui/fashionSigLIP-roturas5: congelar capas y optimizar solo la última + data augmentation.

Similitud con 'roturas-negro1.jpg': 0.9434396624565125
Similitud con 'rotura3.png': 0.9123458862304688
Similitud con 'rotura2.png': 0.9048678874969482
Similitud con 'skinny-rotura.png': 0.8640248775482178
Similitud con 'rotura1.png': 0.8525998592376709
Similitud con 'sin-rotura.png': 0.7889119386672974
📸 Imagen más similar para :roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.9164090156555176
Similitud con 'rotura2.png': 0.8823390603065491
Similitud con 'roturas-negro1.jpg': 0.8589333295822144
Similitud con 'roturas-negro2.jpg': 0.8525997400283813
Similitud con 'skinny-rotura.png': 0.831209659576416
Similitud con 'sin-rotura.png': 0.7981151938438416
📸 Imagen más similar para :rotura1.png: rotura3.png

-----------------------------------------------

melijauregui/fashionSigLIP-roturas6: congelar capas y optimizar últimas 2 + data augmentation.
ds = datasets/roturas-vs-sin.csv

Similitud con 'rotura2.png': 0.9801482558250427
Similitud con 'roturas-negro1.jpg': 0.9769968390464783
Similitud con 'rotura3.png': 0.9749467372894287
Similitud con 'skinny-rotura.png': 0.9723838567733765
Similitud con 'rotura1.png': 0.9637279510498047
Similitud con 'sin-rotura.png': 0.9520493149757385
📸 Imagen más similar para :roturas-negro2.jpg: rotura2.png

Similitud con 'rotura3.png': 0.9840423464775085
Similitud con 'rotura2.png': 0.9737619757652283
Similitud con 'roturas-negro1.jpg': 0.9722333550453186
Similitud con 'skinny-rotura.png': 0.9638115167617798
Similitud con 'roturas-negro2.jpg': 0.963728129863739
Similitud con 'sin-rotura.png': 0.9292023181915283
📸 Imagen más similar para :rotura1.png: rotura3.png

--------------------------------

melijauregui/fashionSigLIP-roturas6: congelar capas y optimizar últimas 2 + data augmentation.
ds = datasets/con-sin-roturas.csv

Similitud con 'roturas-negro1.jpg': 0.9898427724838257
Similitud con 'rotura3.png': 0.9743671417236328
Similitud con 'rotura2.png': 0.9739425778388977
Similitud con 'rotura1.png': 0.9625593423843384
Similitud con 'sin-rotura.png': 0.9488450884819031
Similitud con 'skinny-rotura.png': 0.9397170543670654
📸 Imagen más similar para roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.9862618446350098
Similitud con 'rotura2.png': 0.9846208691596985
Similitud con 'roturas-negro1.jpg': 0.9657573103904724
Similitud con 'roturas-negro2.jpg': 0.9625592231750488
Similitud con 'sin-rotura.png': 0.9616489410400391
Similitud con 'skinny-rotura.png': 0.9606481790542603
📸 Imagen más similar para rotura1.png: rotura3.png

--------------------------

melijauregui/fashionSigLIP-roturas6: congelar capas y optimizar últimas 4 + data augmentation + early stopping.
ds = datasets/con-sin-roturas.csv

Similitud con 'roturas-negro1.jpg': 0.9864325523376465
Similitud con 'rotura3.png': 0.9645108580589294
Similitud con 'rotura2.png': 0.9638651609420776
Similitud con 'sin-rotura.png': 0.9518420696258545
Similitud con 'rotura1.png': 0.951041579246521
Similitud con 'skinny-rotura.png': 0.9225273132324219
📸 Imagen más similar para roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.9843188524246216
Similitud con 'rotura2.png': 0.9816229343414307
Similitud con 'sin-rotura.png': 0.9627786874771118
Similitud con 'skinny-rotura.png': 0.9560232162475586
Similitud con 'roturas-negro2.jpg': 0.9510417580604553
Similitud con 'roturas-negro1.jpg': 0.9506567716598511
📸 Imagen más similar para rotura1.png: rotura3.png

-------------------------

melijauregui/fashionSigLIP-roturas7: congelar capas y optimizar últimas 4 + data augmentation (RANDOM-text) + early stopping.
ds = datasets/con-sin-roturas.csv

Similitud con 'roturas-negro1.jpg': 0.9766131639480591
Similitud con 'rotura2.png': 0.9472299814224243
Similitud con 'rotura3.png': 0.941148579120636
Similitud con 'sin-rotura.png': 0.9390088319778442
Similitud con 'rotura1.png': 0.9366506934165955
Similitud con 'skinny-rotura.png': 0.9147423505783081
📸 Imagen más similar para roturas-negro2.jpg: roturas-negro1.jpg

Similitud con 'rotura3.png': 0.9851783514022827
Similitud con 'rotura2.png': 0.9769280552864075
Similitud con 'skinny-rotura.png': 0.9572533369064331
Similitud con 'sin-rotura.png': 0.9569511413574219
Similitud con 'roturas-negro1.jpg': 0.9450658559799194
Similitud con 'roturas-negro2.jpg': 0.9366506338119507
📸 Imagen más similar para rotura1.png: rotura3.png
"""
