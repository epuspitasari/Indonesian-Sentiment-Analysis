import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Path untuk mengarahkan folder ke direktori src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Cek sinkronisasi
from utils import load_model_artifacts, initial_text_clean, total_clean_pipeline

app = FastAPI(title="Sentiment Analysis API – Good Version", version="2.0")

# Memuat Artefak Model Juara
vectorizer, model = load_model_artifacts("models")

# ============================
# KAMUS SLANG
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

# ============================
# NORMALISASI FRASA SINKRON
# ============================
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

        # Pola DM berpotensi spam/judol
        "dm admin": "dm_admin",
        "dm slot": "dm_slot",
        "dm deposit": "dm_deposit",
        "dm link": "dm_link",
        "dm daftar": "dm_daftar",
        "dm untuk": "dm_untuk",
        "dm wa": "dm_wa"
    }
    # === Cek looping ===
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# ============================
# BLACKLIST CHK
# ============================
DAFTAR_BLACKLIST = {
    "slot", "sl0t", "sl*t", "gacor", "g@cor", "judi", "judol",
    "bandar", "casino", "toto", "bet", "maxwin", "max win",
    "rtp", "rtp tinggi", "deposit", "wd", "wd cepat",
    "link", "alternatif", "iklan", "promosi", "paid-promote",
    "follow", "dm", "hubungi", "hubungi-wa",
    "dm_admin", "dm_slot", "dm_deposit", "dm_link", "dm_daftar", "dm_untuk", "dm_wa"
}

# ============================
# INTERPRETASI INDIKATOR EMOSI
# ============================
NEGASI = {"tidak", "gak", "ga", "nggak", "tak", "tdk"}

KATA_POSITIF = {
    "bagus", "keren", "mantap", "ramah", "terjangkau", "puas",
    "suka", "senang", "nyaman", "aman", "cepat", "bersih",
    "indah", "rekomendasi", "terbaik", "hebat", "murah",
    "membantu", "untung", "lezat", "enak", "top", "mantul",
    "worth", "recommended", "memuaskan", "lebih_baik", "lancar_banget", "lebih_cepat"
}

KATA_NEGATIF = {
    "jelek", "buruk", "kecewa", "mahal", "lambat", "kotor",
    "kasar", "rugi", "nyesel", "kapok", "lelet", "rusak",
    "bohong", "penipu", "kurang", "gagal", "parah", "benci",
    "susah", "sulit", "keluhan", "complaint", "salah", "crash", 
    "payah", "ampas", "hancur", "bau", "berisik", "lemot", "error",
    "mengecewakan", "tidak_bagus", "tidak_sesuai", "lama_banget", 
    "capek_nunggu", "bikin_capek"
}

KELUHAN_KONTEKS = {
    "lama", "antri", "nunggu", "menunggu", "habis", "kehabisan",
    "delay", "macet", "ribet", "capek", "melelahkan", "lama_banget", "capek_nunggu"
}

# ============================
# SCHEMAS (Pydantic)
# ============================
class TextInput(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    source: str

# ============================
# LOGIKA STRATEGIS DETEKSI ATURAN
# ============================
def detect_negation_window(words):
    for i, w in enumerate(words):
        if w in NEGASI:
            window = words[i+1 : i+4]
            if any(next_word in KATA_POSITIF for next_word in window):
                return True
    return False

def detect_sarcasm_raw(text):
    text_lower = text.lower()
    pola = [
        "mantap banget padahal", 
        "bagus banget sampe", 
        "luar biasa antrinya",
        "hebat banget pelayanannya sampai",
        "top banget tapi"
    ]
    return any(p in text_lower for p in pola)

def handle_contrastive_sentences(text_raw):
    text_lower = text_raw.lower()
    kontras_words = ["tapi", "tetapi", "namun", "tpi", "tp"]
    
    for kw in kontras_words:
        if kw in text_lower:
            parts = text_lower.split(kw, 1)
            after_tapi = parts[1]
            
            sinyal_positif_after = ["enak", "bagus", "mantap", "puas", "suka", "nyaman", "worth", "recommended", "memuaskan"]
            if any(pos in after_tapi for pos in sinyal_positif_after):
                return True
    return False


# ============================
# ENDPOINT PREDIKSI SINKRON
# ============================
@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(input_data: TextInput):
    raw = input_data.text.strip()

    # 1. BARIKADE AMAN TEKS KOSONG
    if not raw:
        return {"sentiment": "Netral", "source": "Rule-Based Empty Safe Filter"}

    # 2. BARIKADE DETEKSI SPAM LINK
    if any(x in raw.lower() for x in ["http", "www", ".com", "wa.me"]):
        return {"sentiment": "Tidak Relevan", "source": "Spam-Detected"}

    # 3. BARIKADE DETEKSI SARKASME MENTAH
    if detect_sarcasm_raw(raw):
        return {"sentiment": "Negatif", "source": "Rule-Based Sarcasm Overrule"}

    # PROSES PRA-PEMROSESAN TEKS UTAMA
    cleaned = total_clean_pipeline(raw, SLANG_DICT)
    cleaned = normalize_phrases(cleaned)
    
    words = [w for w in cleaned.split() if w.strip()]
    word_set = set(words)

    # 4. BARIKADE PASCA PREPROCESSING KOSONG
    if not words:
        return {"sentiment": "Netral", "source": "Rule-Based Empty Safe Filter"}

    # 5. BARIKADE FILTER BLACKLIST / NOISE
    if word_set & DAFTAR_BLACKLIST:
        return {"sentiment": "Tidak Relevan", "source": "Noise-Detected"}

    # 6. BARIKADE WINDOW NEGATION
    if detect_negation_window(words):
        return {"sentiment": "Negatif", "source": "Rule-Based Window Negation"}

    # EVALUASI ATURAN DEFENSIVE BERDASARKAN KATA KUNCI
    keluhan = bool(word_set & KELUHAN_KONTEKS)
    punya_neg = bool(word_set & KATA_NEGATIF)
    punya_pos = bool(word_set & KATA_POSITIF)

    # Antisipasi kalimat transisi campuran ("Tapi enak")
    is_contrastive_positive = handle_contrastive_sentences(raw)

    if (punya_neg or keluhan) and is_contrastive_positive:
        # Loloskan ke blok Machine Learning jika ujung kalimat terbukti memuji
        punya_neg = False
        keluhan = False
        punya_pos = True

    # Eksekusi Aturan Ketat
    if punya_neg:
        return {"sentiment": "Negatif", "source": "Rule-Based Strict Negative Guard"}
    if keluhan:
        return {"sentiment": "Negatif", "source": "Rule-Based Complaint Context Guard"}
    
    # Penanganan kalimat pendek netral
    if len(words) <= 2 and not (punya_pos or punya_neg):
        return {"sentiment": "Netral", "source": "Rule-Based Short Text Filter"}

    # Positif murni satu arah
    if punya_pos and not punya_neg:
        return {"sentiment": "Positif", "source": "Rule-Based Pure Positive"}

    # Penanganan teks faktual tanpa emosi indikator
    if not punya_pos and not punya_neg:
        return {"sentiment": "Netral", "source": "Rule-Based Factual Filter"}

    # 7. JALUR EVALUASI UTAMA MODEL ML (LOGISTIC REGRESSION)
    try:
        tfidf = vectorizer.transform([cleaned])
        pred = str(model.predict(tfidf)[0]).strip().lower()

        if pred in ["positive", "positif"]:
            sentiment_val = "Positif"
        elif pred in ["negative", "negatif"]:
            sentiment_val = "Negatif"
        else:
            sentiment_val = "Netral"

        return {
            "sentiment": sentiment_val, 
            "source": "Logistic Regression Model (Mixed Text Decision)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sistem Gagal Memproses: {str(e)}")