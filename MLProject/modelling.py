import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

warnings.filterwarnings('ignore')

print("Script training (modelling.py) v3 dimulai...")

# --- 1. Load Data ---
try:
    df = pd.read_csv("data_bersih.csv")
    print(f"Data berhasil dimuat. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: data_bersih.csv tidak ditemukan!")
    exit()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split selesai.")

# --- 3. Scaling ---
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X) # Transform seluruh data X
print("Scaling data selesai.")

# --- 4. Training Model Terbaik ---
best_params = {
    'max_depth': 20,
    'min_samples_leaf': 4,
    'n_estimators': 100
}
print(f"Melatih model final dengan params: {best_params}")

model = RandomForestRegressor(random_state=42, **best_params)
model.fit(X_scaled, y)
print("Model final selesai dilatih.")

# --- 5. Simpan Artefak (CARA BARU v3 - Memperbaiki TypeError) ---
output_dir = "outputs"
print(f"Menyimpan model ke '{output_dir}' dalam format MLflow...")

# Tentukan signature (skema input/output)
signature = infer_signature(X_scaled, model.predict(X_scaled))

# 1. Simpan model UTAMA. Ini akan MEMBUAT folder 'outputs'
#    dan file 'MLmodel' di dalamnya.
#    Kita HAPUS argumen 'artifacts' yang menyebabkan error.
mlflow.sklearn.save_model(
    sk_model=model,
    path=output_dir,  # Nama folder output
    signature=signature
)

# 2. Sekarang, simpan scaler SECARA MANUAL ke DALAM folder
#    yang baru saja dibuat oleh save_model().
scaler_path = os.path.join(output_dir, "minmax_scaler.joblib")
joblib.dump(scaler, scaler_path)

print(f"Model (di {output_dir}) dan scaler (di {scaler_path}) berhasil disimpan.")
print("Script training selesai.")