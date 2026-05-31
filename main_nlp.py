import os
import sys
import pandas as pd
import joblib  # Digunakan langsung sebagai fallback amannya ekspor artefak
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Tambahkan path src/ agar modul lokal terbaca sempurna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import total_clean_pipeline


def normalize_phrases(text: str) -> str:
    """Normalisasi frasa penting agar model lebih mudah belajar pola sentimen & spam."""
    replacements = {
        # Sentimen umum
        "lebih baik": "lebih_baik",
        "lancar banget": "lancar_banget",
        "lama banget": "lama_banget",
        "capek nunggu": "capek_nunggu",
        "bikin capek": "bikin_capek",
        "lebih cepat": "lebih_cepat",
        "tidak sesuai": "tidak_sesuai",
        "gak sesuai": "tidak_sesuai",
        "tidak bagus": "tidak_bagus",
        "gak bagus": "tidak_bagus",

        # Pola DM berpotensi spam/judol (untuk dipelajari model)
        "dm admin": "dm_admin",
        "dm slot": "dm_slot",
        "dm deposit": "dm_deposit",
        "dm link": "dm_link",
        "dm daftar": "dm_daftar",
        "dm untuk": "dm_untuk",
        "dm wa": "dm_wa"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def main():
    # ============================
    # 1. KAMUS SLANG 
    # ============================
    SLANG_DICT = {
        "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak",
        "nggak": "tidak", "bgt": "banget", "bener": "benar",
        "beneran": "benaran", "krn": "karena", "aj": "saja",
        "aja": "saja", "tp": "tetapi", "tpi": "tapi", "lg": "lagi",
        "udh": "sudah", "udah": "sudah", "blm": "belum",
        "tdk": "tidak", "kalo": "kalau", "kl": "kalau",
        "pke": "pakai", "pake": "pakai", "hub": "hubungi",
        "dm": "direct message", "pm": "private message"
    }

    csv_path = "data/dataset_utama.csv"
    csv_feedback_path = "data/data_koreksi.csv"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # ============================
    # 2. CHECK & LOAD DATASET UTAMA
    # ============================
    print("=== Memuat Dataset Utama ===")
    if not os.path.exists(csv_path):
        print("\n[STANDBY]: Tidak ada dataset utama. Model lama tetap digunakan.\n")
        return

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", skipinitialspace=True)
    df.columns = df.columns.str.strip()

    if "review_text" not in df.columns or "sentiment" not in df.columns:
        print("ERROR: Kolom 'review_text' atau 'sentiment' pada dataset utama tidak ditemukan!")
        return

    # ============================
    # 3. INTEGRASI DATA FEEDBACK & ANTISIPASI DUPLIKASI
    # ============================
    if os.path.exists(csv_feedback_path):
        print("\n[INFO]: File koreksi user ditemukan. Menggabungkan data...")
        try:
            # Menggunakan sep=";" agar sinkron dengan output simpanan app_nlp.py
            df_fb = pd.read_csv(csv_feedback_path, sep=";", encoding="utf-8")
            df_fb.columns = df_fb.columns.str.strip()
            
            # Ubah nama kolom teks_asli -> review_text & prediksi_sistem -> sentiment agar seragam
            df_fb = df_fb.rename(columns={"teks_asli": "review_text", "prediksi_sistem": "sentiment"})
            
            # Ambil kolom yang dibutuhkan saja
            df_fb = df_fb[["review_text", "sentiment"]]
            
            # Gabungkan ke dataframe utama
            df = pd.concat([df, df_fb], ignore_index=True)
            
            # Hapus duplikasi ulasan berdasarkan teks asli, simpan koreksi terbaru dari manusia (keep="last")
            sebelum_drop = len(df)
            df = df.drop_duplicates(subset=["review_text"], keep="last")
            sesudah_drop = len(df)
            
            print(f"Sukses mengasimilasi data koreksi baru. Total baris digabung: {sebelum_drop} -> Baris unik: {sesudah_drop}")
        except Exception as e:
            print(f"[PERINGATAN]: Gagal memproses file koreksi: {e}. Melanjutkan dengan dataset utama.")

    # Pembersihan Data Null / Kosong
    df = df.dropna(subset=["review_text", "sentiment"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]
    
    # Penyelarasan format string teks label sentimen secara seragam
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    
    # Sinkronisasi translasi label bahasa Indonesia ke bahasa Inggris agar sesuai korpus utama model
    mapping_label = {
        "positif": "positive",
        "negatif": "negative",
        "netral": "neutral"
    }
    df["sentiment"] = df["sentiment"].replace(mapping_label)

    print(f"Total baris siap proses: {len(df)}")
    print("Distribusi label saat ini:")
    print(df["sentiment"].value_counts(), "\n")

    # ============================
    # 4. CLEANING + NORMALISASI FRASA
    # ============================
    print("=== Membersihkan Teks ===")
    df["clean_text"] = df["review_text"].apply(lambda x: total_clean_pipeline(x, SLANG_DICT))
    df["clean_text"] = df["clean_text"].apply(normalize_phrases)

    df = df[df["clean_text"].str.strip() != ""]
    if df.empty:
        print("ERROR: Semua teks kosong setelah cleaning!")
        return

    # ============================
    # 5. TF-IDF VECTORIZER
    # ============================
    print("=== TF-IDF Vectorizing ===")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    # ============================
    # 6. SMOTE BALANCING
    # ============================
    print("=== Balancing Data (SMOTE) ===")
    try:
        counts = y.value_counts()
        min_samples = counts.min()

        if min_samples > 1:
            k_val = min(3, min_samples - 1)
            sm = SMOTE(k_neighbors=k_val, random_state=42)
            X_resampled, y_resampled = sm.fit_resample(X, y)
            print(f"SMOTE sukses → Data menjadi {X_resampled.shape[0]} baris.")
        else:
            print("SMOTE dilewati (kelas terlalu sedikit).")
            X_resampled, y_resampled = X, y

    except Exception as e:
        print(f"SMOTE error → dilewati. Alasan: {e}")
        X_resampled, y_resampled = X, y

    # ============================
    # 7. TRAINING MODEL
    # ============================
    print("=== Melatih Model Logistic Regression ===")
    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
        max_iter=1000
    )
    model.fit(X_resampled, y_resampled)

    # ============================
    # 8. EKSPORE ARTIFAK SINKRON
    # ============================
    try:
        # Menggunakan joblib.dump langsung untuk menjamin kompatibilitas bypass jalur pemanggilan pkl
        joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        joblib.dump(model, os.path.join(model_dir, "sentiment_model.pkl"))
        print("\n=== [SELESAI] Artefak Model Baru Berhasil Diperbarui secara Sinkron! ===")
    except Exception as e:
        print(f"ERROR saat menyimpan artefak model: {e}")


if __name__ == "__main__":
    main()