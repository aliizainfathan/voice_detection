import streamlit as st
import librosa
import tsfel
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, dan fitur
model = pickle.load(open("model_gnb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

st.title("Klasifikasi Suara: Buka vs Tutup")

uploaded_file = st.file_uploader("Upload file audio (.mp3)", type=["mp3"])

def extract_features(file):
    cfg = tsfel.get_features_by_domain()
    signal, sr = librosa.load(file, sr=16000)
    fitur = tsfel.time_series_features_extractor(cfg, signal, fs=sr)
    fitur = fitur.reset_index(drop=True)
    fitur = fitur[feature_columns]
    fitur_scaled = scaler.transform(fitur)
    return fitur_scaled

if uploaded_file is not None:
    st.audio(uploaded_file)

    X_new = extract_features(uploaded_file)

    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0]

    label_map = {0: "Buka", 1: "Tutup"}

    st.write("### Hasil Prediksi:")
    st.write(f"**Label:** {label_map[prediction]}")
    st.write(f"**Probabilitas:** {probability}")
