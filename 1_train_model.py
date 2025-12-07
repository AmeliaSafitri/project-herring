# Nama File: 1_train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os

# --- Konfigurasi ---
SERIES_LENGTH = 512
NUM_CLASSES = 2 
MODEL_PATH = './model/cnn_klasifikasi_herring_1d.h5'
TRAIN_DATA_PATH = './data/Herring_TRAIN.ts'
TEST_DATA_PATH = './data/Herring_TEST.ts'

# Fungsi untuk memuat data dari format .ts (UCR Archive)
def load_ts_data(filepath):
    """Membaca data deret waktu dari file .ts."""
    data = []
    labels = []
    
    try:
        with open(filepath, 'r') as f:
            data_started = False
            for line in f:
                line = line.strip()
                if line.startswith('@data'):
                    data_started = True
                    continue
                if not data_started or not line:
                    continue
                
                # Data format: value1,value2,...,valueN:class
                parts = line.split(':')
                if len(parts) == 2:
                    series = np.array([float(x) for x in parts[0].split(',')])
                    # Mengubah label 1 dan 2 menjadi 0 dan 1 (Indeks Kelas)
                    label = int(parts[1]) - 1 
                    data.append(series)
                    labels.append(label)
                    
        X = np.array(data)
        y = np.array(labels)
        
        # Reshape untuk 1D CNN: [sampel, panjang_deret_waktu, 1]
        X = X[..., np.newaxis]
        return X, y
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di jalur: {filepath}")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"ERROR saat memuat data dari {filepath}: {e}")
        return np.array([]), np.array([])


# --- 1. Preprocessing Data ---
print("Memuat dan Memproses Data...")
X_train, y_train_raw = load_ts_data(TRAIN_DATA_PATH)
X_test, y_test_raw = load_ts_data(TEST_DATA_PATH)

if X_train.size == 0 or X_test.size == 0:
    print("Penghentian: Data gagal dimuat. Pastikan file .ts ada di folder ./data/")
    exit()

# Encoding label kelas menjadi format one-hot
y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test_raw, num_classes=NUM_CLASSES)

print(f"Bentuk Data Latih: {X_train.shape}")
print(f"Bentuk Label Latih: {y_train.shape}")
print("-" * 30)

# --- 2. Pemodelan Data (1D CNN) ---
input_shape = X_train.shape[1:] 

def create_1d_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Layer Konvolusi 1D pertama
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        # Layer Konvolusi 1D kedua
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        # Flatten untuk input ke Dense layer
        Flatten(),
        # Fully Connected Layer
        Dense(100, activation='relu'),
        # Output Layer (Softmax untuk klasifikasi multi-kelas)
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_1d_cnn_model(input_shape, NUM_CLASSES)
print("Model 1D CNN dibuat.")

# --- 3. Pelatihan Model ---
print("Memulai pelatihan model...")
# Melatih model menggunakan data latih dan memvalidasi dengan data uji
model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 4. Evaluasi dan Penyimpanan ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("-" * 30)
print(f"✅ Evaluasi Model Selesai:")
print(f"   Akurasi pada Data Uji: {accuracy*100:.2f}%")

# Memastikan direktori model ada dan menyimpan model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"\nModel 1D CNN telah disimpan sebagai: {MODEL_PATH}")