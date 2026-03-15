import pandas as pd
import numpy as np

# 1. CARGAR ARCHIVO CSV

file_name = r"C:\Users\alich\OneDrive\Desktop\Pmicai\TD\traffic.csv"

df = pd.read_csv(file_name)

print("Datos cargados")
print(df.head())

# 2. FECHA Y ORDEN

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values(["sensor_id", "datetime"])

df = df.reset_index(drop=True)

# 3. FUNCION DE TEXTO

def traffic_text(speed):

    if speed < 20:
        return "Severe congestion reported"

    elif speed <= 60:
        return "Moderate traffic flow"

    else:
        return "Free flow traffic"


df["incident_description"] = df["speed"].apply(traffic_text)

# 4. DIFERENCIA POR SENSOR

df["speed_diff"] = df.groupby("sensor_id")["speed"].diff()


def detect_accident(x):

    if pd.isna(x):
        return None

    if abs(x) > 40:
        return "Sudden slowdown possible accident"

    return None


df["accident"] = df["speed_diff"].apply(detect_accident)


df["incident_description"] = np.where(
    df["accident"].notna(),
    df["accident"],
    df["incident_description"]
)

# 5. LIMPIEZA CORRECTA

df = df.dropna(subset=["speed"])

df = df.reset_index(drop=True)

print("DataFrame limpio")
print(df.head())

# 6. DATAFRAME MAESTRO

master_df = df.copy()

# 7. SPLIT TEMPORAL

n = len(master_df)

train_end = int(n * 0.7)
val_end = int(n * 0.85)

train = master_df.iloc[:train_end]
val = master_df.iloc[train_end:val_end]
test = master_df.iloc[val_end:]


print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

# 8. GUARDAR ARCHIVOS

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print("Archivos guardados")

# 9. VERIFICACION

print("Train:")
print(train.head())

print("Val:")
print(val.head())

print("Test:")
print(test.head())