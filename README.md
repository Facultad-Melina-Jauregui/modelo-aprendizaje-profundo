## Requisitos Previos
- Python con entorno virtual configurado en `inference/.venv`
- Variables de entorno y dependencias instaladas (ver `requirements.txt`)

## Ejecución
**Nota:** Todos los comandos deben ejecutarse desde el directorio raíz del proyecto.

### 1. Fine-tuning (Entrenamiento)
```bash
source inference/.venv/bin/activate && \
PYTHONPATH=$(pwd):$PYTHONPATH \
python inference/fashionclipFinetuned/sigLip-finetune-best.py
```
**Archivo ejecutado:** `inference/fashionclipFinetuned/sigLip-finetune-best.py`

### 2. Testing (Pruebas)
Ejecuta los tests del modelo y guarda los resultados:

#### Test de texto-imagen (test.py):
```bash
PROJECT_ROOT=$(pwd) && \
cd inference && \
source .venv/bin/activate && \
PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH \
python fashionclipFinetuned/test.py > resultados-modelo/mejor-modelo/cherrypick-best-sigLip.txt 2>&1
```
**Archivo ejecutado:** `inference/fashionclipFinetuned/test.py`  
**Resultados guardados en:** `inference/resultados-modelo/mejor-modelo/cherrypick-best-sigLip.txt`

#### Test de imagen-imagen (testImage.py):
```bash
PROJECT_ROOT=$(pwd) && \
cd inference && \
source .venv/bin/activate && \
PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH \
python fashionclipFinetuned/testImage.py > resultados-modelo/mejor-modelo/cherrypick-best-sigLip-image.txt 2>&1
```
**Archivo ejecutado:** `inference/fashionclipFinetuned/testImage.py`  
**Resultados guardados en:** `inference/resultados-modelo/mejor-modelo/cherrypick-best-sigLip-image.txt`

## Notas

- Asegúrate de ejecutar primero el fine-tuning antes de correr los tests
- Los resultados del testing se guardan automáticamente en el archivo de salida especificado