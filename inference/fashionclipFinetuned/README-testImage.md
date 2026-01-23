# Tests de Imagen-Imagen (testImage.py)

## Descripción General

El script `testImage.py` permite evaluar modelos fine-tuneados de FashionSigLIP/FashionCLIP mediante pruebas de búsqueda de imágenes similares (imagen-imagen). A diferencia de `test.py` que evalúa texto-imagen, este script compara una imagen query con una galería de imágenes para encontrar las más similares usando embeddings normalizados y similitud coseno.

## Funcionalidad Principal

El script carga un modelo pre-entrenado, procesa imágenes de testing removiendo fondos, y ejecuta pruebas para evaluar el rendimiento del modelo en diferentes categorías:
- **Roturas y tipos de jeans**: wide leg, skinny, recto, palazzo, cargo, mom, flared
- **Prendas generales**: remeras, sweaters, vestidos, camperas, camisas

Cada prueba encuentra la imagen más similar a una imagen query dentro de una galería de imágenes usando embeddings normalizados.

## Sistema de Flags

El sistema utiliza tres tipos de flags para clasificar y evaluar las imágenes:

### Green Flags (✅)
- **Definición**: Imágenes que deberían aparecer primero (exactas/similares)
- **Evaluación**:
  - ✅ Si pasa ambos umbrales (yellow y red) cuando ambos existen
  - 🟡 Si pasa solo el umbral de red (pero no el de yellow) cuando ambos existen
  - ❌ Si no pasa el umbral de red

### Yellow Flags (🟡)
- **Definición**: Imágenes similares pero no exactas
- **Evaluación**:
  - ✅ Si pasa umbral red
  - ❌ Si no pasa el umbral de red

### Red Flags
- **Definición**: Todas las demás imágenes (no green, no yellow)
- **Uso**: Se utiliza el máximo red flag como umbral para evaluar green y yellow flags

## Estructura del Código

### Funciones Principales

#### `find_similarities_image_to_images()`
Calcula similitudes entre una imagen query y una galería de imágenes usando embeddings.

**Parámetros**:
- `model`: Modelo fine-tuneado
- `processor`: Processor del modelo base
- `query_image_path`: Ruta de la imagen query
- `gallery_image_paths`: Lista de rutas de imágenes en la galería
- `gallery_images`: Lista de imágenes PIL ya cargadas
- `green_flags`: Lista de nombres de archivo específicos (exactos)
- `yellow_flags`: Lista de nombres de archivo específicos (similares)

**Proceso**:
1. Genera embeddings normalizados para la query y la galería
2. Calcula similitudes usando producto punto (similitud coseno)
3. Clasifica imágenes según flags
4. Calcula umbrales (máximo yellow flag, máximo red flag)
5. Evalúa y muestra resultados

#### `run_test()`
Ejecuta una prueba individual de búsqueda de imagen similar.

**Parámetros**:
- `query_image_path`: Ruta de la imagen query
- `image_paths`: Lista de rutas de imágenes en la galería
- `images`: Lista de imágenes PIL ya cargadas
- `green_flags`: Lista de nombres de archivo específicos (exactos)
- `yellow_flags`: Lista de nombres de archivo específicos (similares)
- `test`: Si es True, muestra los flags

#### `run_tests()`
Procesa imágenes de una carpeta y ejecuta una función de pruebas.

**Proceso**:
1. Remueve fondos de las imágenes usando `remove_background()`
2. Carga todas las imágenes procesadas
3. Ejecuta la función de pruebas con las imágenes cargadas

#### `run_tests_queries()`
Función genérica que ejecuta tests para múltiples queries.

**Parámetros**:
- `image_paths`: Lista de rutas de imágenes
- `images`: Lista de imágenes PIL
- `test`: Flag de testing
- `query_candidates`: Lista de nombres de archivo de queries
- `get_queries_with_flags_func`: Función que retorna los flags para cada query

## Configuración de Queries y Flags

### Definir Queries

Las queries se definen como listas de nombres de archivo en las funciones específicas:

```python
query_candidates = [
    'wide-rotura-simetrica-negro1.png',
    'wide-rotura-simetrica1.png',
    'skinny-rotura-simetrica2.png',
    # ...
]
```

### Definir Flags

Los flags se definen en funciones específicas que retornan un diccionario con `green` y `yellow`:

```python
def get_queries_with_flags_roturas(query_path):
    query_candidates = {
        'wide-rotura-simetrica1.png': {
            'green': ['wide-rotura-simetrica2.png', 'wide-rotura-simetrica-desmechada1.png'],
            'yellow': ['recto-rotura-simetrica1.png', 'wide1.png']
        },
        # ...
    }
    return query_candidates[query_path]
```

**Importante**: 
- Los green flags están ordenados por prioridad (el primero debería ser más similar)
- Los yellow flags también están ordenados por prioridad
- Los nombres de archivo deben coincidir exactamente con los archivos en la galería

## Métricas Calculadas

### Accuracy
- **Cálculo**: Porcentaje de green flags que pasan ambos umbrales (yellow y red)
- **Umbral**: >= 0.6 (60%)
- **Fórmula**: `correct / total * 100`

### Accuracy Aproximada
- **Cálculo**: Porcentaje de green flags que pasan al menos el umbral de red (incluye 🟡)
- **Se muestra**: Solo si es mayor que la accuracy normal

### Precision
- **Cálculo**: TP / (TP + FP)
- **Umbral**: >= 0.7 (70%)
- **TP (True Positives)**: Green flags y yellow flags que pasan el threshold
- **FP (False Positives)**: Imágenes red que pasan el threshold yellow pero no son green ni yellow

### Recall
- **Cálculo**: TP / (TP + FN)
- **Umbral**: >= 0.6 (60%)
- **FN (False Negatives)**: Green flags y yellow flags que no pasan el threshold red

### MRR (Mean Reciprocal Rank)
- **Cálculo**: 1 / rank del primer green flag en el ranking
- **Umbral**: >= 0.3
- **Rank**: Posición del primer green flag cuando se ordenan todas las imágenes por similitud

### Top-3 Check
- **Verificación**: Si el primer green flag aparece en las primeras 3 posiciones del ranking
- **Umbral**: rank <= 3

## Salida del Script

Para cada query, el script muestra:

1. **Información de umbrales**:
   ```
   mayor red flag es [nombre] con [similitud] de similitud
   mayor yellow flag es [nombre] con [similitud] de similitud
   ```

2. **Green flags** (en orden de prioridad):
   ```
   green_flags:
     ✅ [nombre]: [similitud]
     🟡 [nombre]: [similitud]
     ❌ [nombre]: [similitud]
   ```

3. **Yellow flags** (en orden de prioridad):
   ```
   yellow_flags:
     ✅ [nombre]: [similitud]
     ❌ [nombre]: [similitud]
   ```

4. **Métricas**:
   ```
   🎯 Porcentaje de acierto: [%] ([correct]/[total])
   📊 Precision: [valor]
   📊 Recall: [valor]
   📊 MRR (Mean Reciprocal Rank): [valor]
   📊 Primer green flag aparece en rank: [rank]
   ```

5. **Resumen final**:
   ```
   RESUMEN FINAL:
   ✅/❌ Accuracy vs red/yellow flags (>= 0.6)
   ✅/❌ Precision (>= 0.7)
   ✅/❌ Recall (>= 0.6)
   ✅/❌ MRR (>= 0.3)
   ✅/❌ Top-3 check
   ==============================
   ```

## Uso

### Ejecutar Tests de Roturas

```bash
cd inference
source .venv/bin/activate
PYTHONPATH=/Users/melinajauregui/Documents/modeloAprendizajeProfundo:$PYTHONPATH python fashionclipFinetuned/testImage.py
```

El script ejecutará automáticamente:
- Tests de roturas (si están habilitados en `__main__`)
- Tests de prendas generales (si están habilitados en `__main__`)

### Redirigir Salida a Archivo

```bash
python fashionclipFinetuned/testImage.py > resultados-modelo/mejor-modelo/cherrypick-best-sigLip-image.txt 2>&1
```

## Estructura de Carpetas

- **`images-testing-roturas/`**: Imágenes originales de testing de roturas
- **`images-testing-roturas-nobg/`**: Imágenes procesadas (sin fondo) - se genera automáticamente
- **`images_testing_general/`**: Imágenes originales de testing de prendas generales
- **`images-testing-general-nobg/`**: Imágenes procesadas (sin fondo) - se genera automáticamente

## Notas Importantes

1. **Procesamiento de imágenes**: El script automáticamente remueve fondos de las imágenes antes de procesarlas usando `rm_bg.remove_background()`. Las imágenes procesadas se guardan en formato PNG.

2. **Exclusión de query**: La imagen query se excluye automáticamente de todos los cálculos (no se evalúa contra sí misma).

3. **Orden de prioridad**: Los green y yellow flags se evalúan en el orden especificado en las listas, manteniendo la prioridad definida.

4. **Umbrales dinámicos**: Los umbrales se calculan dinámicamente:
   - `max_red_flag`: Máxima similitud de todas las imágenes red
   - `max_yellow_flag`: Máxima similitud de todas las imágenes yellow
   - `min_yellow_flag`: Mínima similitud de todas las imágenes yellow (para cálculo de precision)

5. **Formato de archivos**: Las imágenes deben estar en formato PNG, JPG o JPEG. Después del procesamiento, todas se convierten a PNG.

## Personalización

Para agregar nuevas queries o modificar flags:

1. **Agregar query a la lista** en `run_tests_roturas()` o `run_tests_general()`
2. **Definir flags** en `get_queries_with_flags_roturas()` o `get_queries_with_flags_general()`
3. **Asegurar que los nombres de archivo coincidan** exactamente con los archivos en la galería

## Ejemplo de Configuración

```python
def get_queries_with_flags_roturas(query_path):
    query_candidates = {
        'wide-rotura-simetrica1.png': {
            'green': [
                'wide-rotura-simetrica2.png',           # Más similar
                'wide-rotura-simetrica-desmechada1.png', # Menos similar
                'recto-rotura4.png'                      # Aún menos similar
            ],
            'yellow': [
                'recto-rotura-simetrica1.png',  # Similar pero diferente tipo
                'wide1.png',                     # Similar pero sin roturas
                'palazzo2.png'                   # Similar pero diferente corte
            ]
        },
    }
    return query_candidates[query_path]
```

En este ejemplo, `wide-rotura-simetrica2.png` debería ser más similar que `wide-rotura-simetrica-desmechada1.png`, y así sucesivamente.

---

## Resumen de Resultados (cherrypick-best-sigLip)

Resumen y promedios de métricas sobre **17 queries** (10 roturas + 7 prendas generales), tomados de `resultados-modelo/mejor-modelo/cherrypick-best-sigLip-image.txt`.

Las métricas se calculan con `parse_test_metrics.py` (parseo de los archivos de resultados).

### Promedios por query

| Métrica | Promedio | Mín. | Máx. |
|--------|----------|------|------|
| **Accuracy** (green flags %) | **92.2%** | 50% | 100% |
| **Precision** | **0.79** | 0.18 | 1.00 |
| **Recall** | **0.83** | 0.40 | 1.00 |
| **MRR** | **0.50** | 0.50 | 0.50 |

### Cumplimiento de umbrales (por query)

| Umbral | ✅ Cumple | ❌ No cumple | % Cumple |
|--------|-----------|--------------|----------|
| Accuracy ≥ 0.6 | 15 | 2 | **88.2%** |
| Precision ≥ 0.7 | 12 | 5 | **70.6%** |
| Recall ≥ 0.6 | 15 | 2 | **88.2%** |
| MRR ≥ 0.3 | 17 | 0 | **100%** |
| Top-3 check | 17 | 0 | **100%** |

### Queries con métricas por debajo del umbral

- **Accuracy &lt; 0.6**: `recto-rotura-desmechada1.png` (50%), `sweater-rosa.png` (50%).
- **Precision &lt; 0.7**: `recto-rotura-desmechada1.png`, `skinny1.png`, `campera-cuero-negra.png`, `campera-jean-azul.png`, `remera-celeste1.png`.
- **Recall &lt; 0.6**: `skinny1.png`, `campera-jean-azul.png`.

### Desglose por categoría

| Categoría | Queries | Accuracy prom. | Precision prom. | Recall prom. |
|-----------|---------|----------------|-----------------|--------------|
| **Roturas** | 10 | 91.7% | 0.86 | 0.89 |
| **Prendas generales** | 7 | 92.9% | 0.69 | 0.75 |

*Nota: los promedios se obtienen promediando las métricas por query. Para actualizar este resumen, ejecutar `python parse_test_metrics.py` y revisar el output, o volver a correr los tests y luego el script.*
