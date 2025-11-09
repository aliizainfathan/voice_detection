import streamlit as st
import librosa
import tsfel
import numpy as np
import pandas as pd
import pickle
import io # Diperlukan untuk menangani audio-bytes
from audio_recorder_streamlit import audio_recorder # Import komponen perekam

# --- TAHAP 1: MUAT SEMUA MODEL DENGAN CACHE ---
# Menggunakan cache agar model hanya di-load sekali
model_knn = pickle.load(open("model_knn.pkl", "rb"))
scaler_knn = pickle.load(open("scaler_knn.pkl", "rb"))
model_person1 = pickle.load(open("knn_person1.pkl", "rb"))
scaler_knn_person1 = pickle.load(open("scaler_knn_person1.pkl", "rb"))
model_person2 = pickle.load(open("knn_person2.pkl", "rb"))
scaler_knn_person2 = pickle.load(open("scaler_knn_person2.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# TAHAP 2: FUNGSI EKSTRAKSI FITUR 

def extract_features(file, feature_list):
    cfg = tsfel.get_features_by_domain()
    signal, sr = librosa.load(file, sr=16000)
    fitur_raw = tsfel.time_series_features_extractor(cfg, signal, fs=sr)
    fitur_raw = fitur_raw.reset_index(drop=True)
    # Pilih hanya fitur yang kita gunakan saat latihan
    # Penting: Pastikan urutan kolom sama
    X_new = fitur_raw[feature_list]
    return X_new

# TATA LETAK STREAMLIT ---
st.title("Voice Detection")
st.write("Aplikasi ini hanya mengenali suara dua orang saja.")

# Sidebar (Pengaturan Threshold) - Kode Anda sudah benar
st.sidebar.header("Pengaturan Threshold")
speaker_thresh = st.sidebar.slider(
    "Threshold Keyakinan Speaker", 
    min_value=0.5, 
    max_value=1.0, 
    value=0.80, 
    step=0.01,
    help="Ambang batas minimum probabilitas untuk mengenali speaker (Fathan/Zanuar)."
)
command_thresh = st.sidebar.slider(
    "Threshold Keyakinan Perintah", 
    min_value=0.5, 
    max_value=1.0, 
    value=0.70, 
    step=0.01,
    help="Ambang batas minimum probabilitas untuk mengenali perintah (Buka/Tutup)."
)

# Input Audio 
tab1, tab2 = st.tabs(["Upload File Audio", "Rekam Suara Langsung"])
audio_source = None # Variabel untuk menampung audio

with tab1:
    uploaded_file = st.file_uploader(
        "Upload file audio (.mp3, .wav)", 
        type=["mp3", "wav"], 
        key="uploader"
    )
    if uploaded_file:
        audio_source = uploaded_file 

with tab2:
    st.write("Tekan tombol di bawah untuk mulai/berhenti merekam:")
    audio_bytes = audio_recorder(
        text="Tekan untuk Merekam",
        icon_size="2x",
        key="recorder"
    )
    if audio_bytes:
        audio_source = io.BytesIO(audio_bytes) 

if audio_source is not None:
    # Tampilkan pemutar audio untuk file yang dipilih
    st.audio(audio_source) 
    
    if st.button("Analisis Suara"):
        with st.spinner("Mengekstrak fitur dan menganalisis..."):
            
            # 1. Ekstrak Fitur (sekarang menggunakan 'audio_source')
            X_new = extract_features(audio_source, feature_columns)
            
            if X_new is not None:
                # Verifikasi Speaker
                X_fathan_scaled = scaler_knn_person1.transform(X_new)
                prob_fathan = model_person1.predict_proba(X_fathan_scaled)[0][1]

                X_zanuar_scaled = scaler_knn_person2.transform(X_new)
                prob_zanuar = model_person2.predict_proba(X_zanuar_scaled)[0][1]

                identified_speaker = "tidak dikenali"
                
                if prob_fathan >= speaker_thresh:
                    identified_speaker = "Fathan"
                elif prob_zanuar >= speaker_thresh:
                    identified_speaker = "Zanuar"

                # Tampilkan Hasil Speaker
                if identified_speaker == "tidak dikenali":
                    st.error(f"🔴 AKSES DITOLAK! Speaker tidak dikenali.")
                    with st.expander("Lihat Detail Probabilitas Speaker"):
                        st.write(f"Probabilitas Fathan: {prob_fathan:.2%}")
                        st.write(f"Probabilitas Zanuar: {prob_zanuar:.2%}")
                        st.write(f"(Threshold: {speaker_thresh:.2%})")
                
                else:
                    st.success(f"🟢 AKSES DITERIMA! Speaker teridentifikasi: **{identified_speaker}**")
                    
                    # === Prediksi Perintah (Hanya jika speaker lolos) ===
                    X_command_scaled = scaler_knn.transform(X_new)
                    pred_num = model_knn.predict(X_command_scaled)[0]
                    prob_command = model_knn.predict_proba(X_command_scaled)[0]
                    max_prob_command = np.max(prob_command)

                    label_map = {0: "Buka", 1: "Tutup"}
                    pred_label = label_map.get(pred_num, "Error")

                    if max_prob_command < command_thresh:
                        st.warning(f"Perintah tidak dikenali (Keyakinan: {max_prob_command:.2%})")
                    else:
                        st.write("--- Hasil Perintah ---")
                        st.subheader(f"Prediksi Perintah: **{pred_label}**")
                        st.write(f"Keyakinan Perintah: {max_prob_command:.2%}")