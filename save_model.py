# Simpan model
with open("model_gnb.pkl", "wb") as f:
    pickle.dump(gnb, f)

# simpan scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# simpan nama fitur (penting agar test datanya sama urutannya)
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

print("✅ Model berhasil disimpan")