"""
Script para balancear el dataset de prendas con y sin roturas.

Este script lee un dataset de prendas, identifica los pantalones tipo "wide leg",
y balancea la cantidad de muestras con y sin roturas para esta categoría específica,
limitando cada subcategoría a un máximo de 25 muestras. El resto de las prendas
se mantiene sin cambios.
"""
import pandas as pd
import random

# Leer el dataset
df = pd.read_csv('datasets/con-sin-roturas-v3.csv')

# Preprocesar
df['description_lower'] = df['description'].str.lower()


def detectar_corte(desc):
    if 'jean wide leg' in desc:
        return 'wide leg'
    return None

# Función para saber si tiene roturas


def tiene_roturas(desc):
    return 'rotura' in desc


# Clasificar las imágenes
df['corte'] = df['description_lower'].apply(detectar_corte)
df['rotura'] = df['description_lower'].apply(tiene_roturas)

# Separar wide leg
wide_leg = df[df['corte'] == 'wide leg']

# Wide leg con rotura
wl_con_rotura = wide_leg[wide_leg['rotura'] == True]
if len(wl_con_rotura) > 25:
    wl_con_rotura = wl_con_rotura.sample(25, random_state=42)

# Wide leg sin rotura
wl_sin_rotura = wide_leg[wide_leg['rotura'] == False]
if len(wl_sin_rotura) > 25:
    wl_sin_rotura = wl_sin_rotura.sample(25, random_state=42)

# Todo el resto (que no es wide leg)
resto = df[df['corte'] != 'wide leg']

# Concatenar todo
df_final = pd.concat([resto, wl_con_rotura, wl_sin_rotura])

# Limpiar columnas auxiliares
df_final = df_final.drop(columns=['description_lower', 'corte', 'rotura'])

# Guardar el nuevo dataset
df_final.to_csv('datasets/con-sin-roturas-v4.csv', index=False)

print(f"Nuevo dataset generado con {len(df_final)} registros.")
