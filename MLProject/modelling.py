import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings('ignore')

print("Script training (modelling.py) dimulai...")

# --- 1. Load Data -----
try:
    df = pd.read_csv("data_bersih.csv")
    print(f"Data berhasil dimuat. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: data_bersih.csv tidak ditemukan!")
    exit()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# --- 2. Split Data ---
# Kita split hanya untuk scaling, tapi kita akan latih di SEMUA dataa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split selesai.")

# --- 3. Scaling ---
scaler = MinMaxScaler()
# Fit scaler di data training
scaler.fit(X_train)
# Transform seluruh data X
X_scaled = scaler.transform(X)
print("Scaling data selesai.")

# --- 4. Training Model Terbaik ---
# Gunakan parameter terbaik dari Kriteria 2
best_params = {
    'max_depth': 20,
    'min_samples_leaf': 4,
    'n_estimators': 100
}
print(f"Melatih model final dengan params: {best_params}")

model = RandomForestRegressor(random_state=42, **best_params)
# Latih model di SEMUA data yang sudah di-scale
model.fit(X_scaled, y)
print("Model final selesai dilatih.")

# --- 5. Simpan Artefak (Model dan Scaler) ---
# Ini adalah output dari MLProject kita
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "rf_model.joblib")
scaler_path = os.path.join(output_dir, "minmax_scaler.joblib")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Model tersimpan di: {model_path}")
print(f"Scaler tersimpan di: {scaler_path}")
print("Script training selesai.")