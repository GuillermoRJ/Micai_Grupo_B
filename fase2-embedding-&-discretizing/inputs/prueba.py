import pandas as pd
import os
import torch
import math
from transformers import RobertaTokenizer, RobertaModel

# --- 1. CONFIGURACIÓN DE RUTAS Y DISPOSITIVO ---
directorio_script = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(directorio_script)
directorio_outputs = os.path.join(directorio_raiz, 'outputs')

ruta_entrada = os.path.join(directorio_script, 'tabla_maestra.csv')
ruta_vectores = os.path.join(directorio_outputs, 'vectores_accidentes.pt')
ruta_solo_velocidades = os.path.join(directorio_outputs, 'categoria_trafico.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- 2. CARGA DE DATOS ---
if not os.path.exists(ruta_entrada):
    print(f"Error: No se encontró el archivo {ruta_entrada}")
    exit()
k
df = pd.read_csv(ruta_entrada)
print("Datos cargados correctamente.")

# --- 3. PROCESAMIENTO DE TEXTO (ROBERTA) ---
frases_accidentes = df["incident_text"].dropna().tolist()
print(f"Total de frases a procesar: {len(frases_accidentes)}")

print("Cargando el modelo RoBERTa...")
nombre_modelo = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(nombre_modelo)
model = RobertaModel.from_pretrained(nombre_modelo).to(device)
model.eval()

batch_size = 32
num_batches = math.ceil(len(frases_accidentes) / batch_size)
todos_los_vectores = []

print("Comenzando el procesamiento de embeddings por lotes...")
with torch.no_grad():
    for i in range(0, len(frases_accidentes), batch_size):
        lote_frases = frases_accidentes[i : i + batch_size]
        
        inputs = tokenizer(
            lote_frases,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        outputs = model(**inputs)
        # Extraer token [CLS]
        vectores_lote = outputs.last_hidden_state[:, 0, :].cpu()
        todos_los_vectores.append(vectores_lote)

        if ((i // batch_size) + 1) % 10 == 0 or ((i // batch_size) + 1) == num_batches:
            print(f"Procesado lote {(i // batch_size) + 1} de {num_batches}...")

# Guardar vectores
tensor_final = torch.cat(todos_los_vectores, dim=0)
torch.save(tensor_final, ruta_vectores)
print(f"¡Vectores guardados en: {ruta_vectores}")


# --- 4. PROCESAMIENTO DE VELOCIDADES (UNIFICADO) ---
print("\nProcesando categorías de velocidad...")

# Cálculo de km/h
df['speed_kmh'] = df['speed_mph'] * 1.60934

# Definición de categorías
limites_bins = [0, 10, 25, 40, 65, float('inf')]
etiquetas_categorias = [
    'Congestión severa', # 0 a 10
    'Tráfico lento',     # 11 a 25
    'Tráfico moderado',  # 26 a 40
    'Tráfico fluido',    # 41 a 65
    'Vías rápidas'       # Más de 65
]

# Categorización
df['categoria_trafico'] = pd.cut(
    df['speed_kmh'], 
    bins=limites_bins, 
    labels=etiquetas_categorias, 
    right=True
)

# --- 5. GUARDAR SOLO LA INFORMACIÓN DE VELOCIDAD ---
# Creamos un nuevo DataFrame solo con las columnas deseadas
df_solo_velocidades = df[['speed_mph', 'speed_kmh', 'categoria_trafico']].copy()

# Guardar en un archivo nuevo
df_solo_velocidades.to_csv(ruta_solo_velocidades, index=False, encoding='utf-8')

print(f"✅ ¡Proceso de velocidades finalizado!")
print(f"El archivo con la información categorizada se ha guardado en:\n{ruta_solo_velocidades}")

# Mostrar muestra final por consola
print("\nMuestra del nuevo archivo de velocidades:")
print(df_solo_velocidades.head(10))