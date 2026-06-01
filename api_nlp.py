import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import load_model_artifacts, total_clean_pipeline

app = FastAPI(title="Sentiment Analysis API – Sinkron Version", version="2.0")

try:
    vectorizer, model = load_model_artifacts()
except Exception:
    import joblib
    try:
        vectorizer = joblib.load(os.path.join("models", "tfidf_vectorizer.pkl"))
        model = joblib.load(os.path.join("models", "sentiment_model.pkl"))
    except Exception as e:
        vectorizer, model = None, None
        print(f"[ERROR] Gagal memuat model: {e}")

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

def normalize_phrases(text: str) -> str:
    replacements = {
        "lebih baik": "lebih_baik", "lancar banget": "lancar_banget",
        "lama banget": "lama_banget", "capek nunggu": "capek_nunggu",
        "bikin capek": "bikin_capek", "lebih cepat": "lebih_cepat",
        "tidak sesuai": "tidak_sesuai", "gak sesuai": "tidak_sesuai",
        "tidak bagus": "tidak_bagus", "gak bagus": "tidak_bagus",
        "dm admin": "dm_admin", "dm slot": "dm_slot", "dm deposit": "dm_deposit",
        "dm link": "dm_link", "dm daftar": "dm_daftar", "dm untuk": "dm_untuk", "dm wa": "dm_wa"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

DAFTAR_BLACKLIST = {
    "slot", "sl0t", "sl*t", "gacor", "g@cor", "judi", "judol",
    "bandar", "casino", "toto", "bet", "maxwin", "max win",
    "rtp", "rtp tinggi", "deposit", "wd", "wd cepat",
    "link", "alternatif", "iklan", "promosi", "paid-promote",
    "follow", "dm", "hubungi", "hubungi-wa",
    "dm_admin", "dm_slot", "dm_deposit", "dm_link", "dm_daftar", "dm_untuk", "dm_wa"
}

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

class TextInput(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    source: str

def detect_negation_window(words):
    for i, w in enumerate(words):
        if w in NEGASI:
            window = words[i+1 : i+4]
            if any(next_word in KATA_POSITIF for next_word in window):
                return True
    return False

def detect_double_negation(words):
    for i, w in enumerate(words):
        if w in NEGASI:
            window = words[i+1 : i+3]
            if any(next_word in KATA_NEGATIF for next_word in window):
                return True
    return False

def detect_sarcasm_raw(text):
    text_lower = text.lower()
    pola = ["mantap banget padahal", "bagus banget sampe", "luar biasa antrinya", "hebat banget pelayanannya sampai", "top banget tapi"]
    return any(p in text_lower for p in pola)

def handle_contrastive_sentences(text_raw):
    text_lower = text_raw.lower()
    kontras_words = ["tapi", "tetapi", "namun", "tpi", "tp"]
    for kw in kontras_words:
        if kw in text_lower:
            parts = text_lower.split(kw, 1)
            if len(parts) > 1:
                after_tapi = parts[1]
                words_after = set([w.strip() for w in after_tapi.split() if w.strip()])
                sinyal_positif_after = {"enak", "bagus", "mantap", "puas", "suka", "nyaman", "worth", "recommended", "memuaskan"}
                if words_after & sinyal_positif_after:
                    return True
    return False

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(input_data: TextInput):
    raw = input_data.text.strip()

    if not raw:
        return {"sentiment": "Netral", "source": "Rule-Based Empty Safe Filter"}

    if any(x in raw.lower() for x in ["http", "www", ".com", "wa.me"]):
        return {"sentiment": "Tidak Relevan", "source": "Spam-Detected"}

    if detect_sarcasm_raw(raw):
        return {"sentiment": "Negatif", "source": "Rule-Based Sarcasm Overrule"}

    cleaned = total_clean_pipeline(raw, SLANG_DICT)
    cleaned = normalize_phrases(cleaned)
    
    words = [w for w in cleaned.split() if w.strip()]
    word_set = set(words)

    if not words:
        return {"sentiment": "Netral", "source": "Rule-Based Empty Safe Filter"}

    if word_set & DAFTAR_BLACKLIST:
        return {"sentiment": "Tidak Relevan", "source": "Noise-Detected"}

    if detect_negation_window(words):
        return {"sentiment": "Negatif", "source": "Rule-Based Window Negation"}

    keluhan = bool(word_set & KELUHAN_KONTEKS)
    punya_neg = bool(word_set & KATA_NEGATIF)
    punya_pos = bool(word_set & KATA_POSITIF)

    if detect_double_negation(words):
        punya_neg = False
        punya_pos = True

    is_contrastive_positive = handle_contrastive_sentences(raw)

    if (punya_neg or keluhan) and is_contrastive_positive:
        punya_neg = False
        keluhan = False
        punya_pos = True

    if { "tidak_bagus", "tidak_sesuai" } & word_set:
        punya_neg = True

    if punya_neg:
        return {"sentiment": "Negatif", "source": "Rule-Based Strict Negative Guard"}
    if keluhan:
        return {"sentiment": "Negatif", "source": "Rule-Based Complaint Context Guard"}
    if len(words) <= 2 and not (punya_pos or punya_neg):
        return {"sentiment": "Netral", "source": "Rule-Based Short Text Filter"}
    if punya_pos and not punya_neg:
        return {"sentiment": "Positif", "source": "Rule-Based Pure Positive"}
    if not punya_pos and not punya_neg:
        return {"sentiment": "Netral", "source": "Rule-Based Factual Filter"}

    if vectorizer is None or model is None:
        return {"sentiment": "Error", "source": "Model ML pkl gagal dimuat"}

    try:
        tfidf = vectorizer.transform([cleaned])
        pred = str(model.predict(tfidf)[0]).strip().lower()
        sentiment_val = "Positif" if pred in ["positive", "positif"] else ("Negatif" if pred in ["negative", "negatif"] else "Netral")
        return {"sentiment": sentiment_val, "source": "Logistic Regression Model (Mixed Text Decision)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sistem Gagal Memproses: {str(e)}")