# Nama File: app_1d.py
import streamlit as st
import tensorflow as tf
import numpy as np
import io
import os

# --- Konfigurasi Model & Kelas ---
MODEL_PATH = './model/cnn_klasifikasi_herring_1d.h5' 
SERIES_LENGTH = 304
CLASS_NAMES = ['North Sea (Kelas 1)', 'Thames (Kelas 2)'] 
CONFIDENCE_THRESHOLD = 0.80 # Batas kepercayaan (80%)

# Fungsi untuk memuat model (Menggunakan st.cache_resource agar hanya dimuat sekali)
@st.cache_resource
def load_1d_model():
    """Memuat model 1D CNN yang sudah dilatih."""
    try:
        # Pengecekan jalur file
        if not os.path.exists(MODEL_PATH):
            st.error(f"⚠️ File model tidak ditemukan di: {MODEL_PATH}. Harap jalankan '1_train_model.py' terlebih dahulu.")
            return None
            
        model = tf.keras.models.load_model(MODEL_PATH) 
        return model
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model. Error: {e}")
        return None

# Fungsi untuk preprocessing data deret waktu yang diunggah (VERSI YANG DIPERBAIKI)
def preprocess_series_data(uploaded_file):
    """Membaca file dan menyiapkan data untuk prediksi 1D CNN. Menangani pemisah koma atau spasi."""
    try:
        # 1. Membaca konten file
        data_string = uploaded_file.getvalue().decode("utf-8").strip()
        
        # 2. Mengurai data (menghapus label kelas jika ada, misal ':1' di akhir)
        if ':' in data_string:
            data_string = data_string.split(':')[0]
            
        # 3. PENANGANAN PEMISAH SPASI/KOMA (Perbaikan Error)
        # Menghapus spasi berlebihan dan mengganti semua spasi dengan koma untuk menyeragamkan pemisah
        data_string = ' '.join(data_string.split()).replace(' ', ',')
        # Hapus koma berlebihan di awal/akhir jika ada
        data_string = data_string.strip(', ')

        # 4. Mengubah string deret waktu menjadi array float
        # Gunakan list comprehension untuk mengkonversi nilai (mengabaikan string kosong jika ada)
        series = np.array([float(x) for x in data_string.split(',') if x])
        
        if len(series) != SERIES_LENGTH:
            st.error(f"❌ Kesalahan: Panjang deret waktu harus {SERIES_LENGTH}. Ditemukan {len(series)}.")
            return None
            
        # 5. Reshape untuk 1D CNN: [1, panjang_deret_waktu, 1]
        series = series.reshape(1, SERIES_LENGTH, 1)
        
        return series
    except Exception as e:
        st.error(f"❌ Kesalahan saat memproses file: Pastikan formatnya adalah {SERIES_LENGTH} angka float/int dipisahkan koma atau spasi. Detail: {e}")
        return None

# ===============================================
#          APLIKASI STREAMLIT UTAMA
# ===============================================

st.set_page_config(page_title="Herring Otolith Classifier", layout="centered")

st.title("🐟 Aplikasi Klasifikasi Otolith Herring (1D CNN)")
st.subheader(f"Klasifikasi berdasarkan Deret Waktu ke **{CLASS_NAMES[0]}** atau **{CLASS_NAMES[1]}**")

model = load_1d_model()

if model is not None:
    uploaded_file = st.file_uploader(
        "Unggah File Deret Waktu Otolith Sampel Tunggal (TXT, TS, atau CSV)", 
        type=["txt", "ts", "csv"]
    )

    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("Hasil Klasifikasi")
        
        # Lakukan preprocessing
        series_array = preprocess_series_data(uploaded_file)
        
        if series_array is not None:
            # Lakukan prediksi
            with st.spinner('⏳ Menganalisis deret waktu...'):
                try:
                    probabilities = model.predict(series_array)[0]
                    
                    predicted_class_index = np.argmax(probabilities)
                    predicted_class_name = CLASS_NAMES[predicted_class_index]
                    confidence = probabilities[predicted_class_index]
                    
                    # --- LOGIKA PESAN "DATA TIDAK DIKENALI" ---
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.error("❌ **DATA TIDAK DIKENALI**")
                        st.warning(f"Model tidak yakin dengan sampel ini.")
                        st.info(f"Tingkat kepercayaan tertinggi hanya {confidence:.2%}, di bawah batas **{CONFIDENCE_THRESHOLD:.0%}**.")
                    else:
                        st.success(f"✅ Data diklasifikasikan sebagai: **{predicted_class_name}**")
                        st.balloons()
                        st.info(f"Tingkat Kepercayaan (Confidence): **{confidence:.2%}**")
                        
                    st.markdown("##### Probabilitas per Kelas:")
                    for name, prob in zip(CLASS_NAMES, probabilities):
                        st.write(f"- **{name}**: {prob:.2%}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
                
st.markdown("---")
st.caption("Pastikan Anda telah menjalankan `1_train_model.py` untuk melatih model.")