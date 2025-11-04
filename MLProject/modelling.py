import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
import mlflow  # <-- Import MLflow
import mlflow.sklearn
from mlflow.models import infer_signature  # <-- Import untuk signature

warnings.filterwarnings('ignore')

print("Script training (modelling.py) v2 dimulai...")

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
y_pred_for_signature = scaler.transform(X_test) # Hanya untuk signature
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

# --- 5. Simpan Artefak (CARA BARU DENGAN MLFLOW) ---
output_dir = "outputs"
print(f"Menyimpan model ke '{output_dir}' dalam format MLflow...")

# Simpan scaler ke file sementara agar bisa di-bundel
scaler_temp_path = "minmax_scaler.joblib"
joblib.dump(scaler, scaler_temp_path)

# Tentukan signature (skema input/output) untuk model
# Ini penting untuk 'mlflow models build-docker'
signature = infer_signature(X_scaled, model.predict(X_scaled))

# Simpan model menggunakan MLflow
# Ini akan membuat folder 'outputs/' LENGKAP dengan file 'MLmodel'
mlflow.sklearn.save_model(
    sk_model=model,
    path=output_dir,  # Nama folder output
    signature=signature,
    artifacts={
        "scaler": scaler_temp_path  # Bundel scaler sebagai artefak
    }
)

# Hapus file scaler sementara
os.remove(scaler_temp_path)

print(f"Model dan scaler berhasil disimpan di folder: {output_dir}")
print("Script training selesai.")