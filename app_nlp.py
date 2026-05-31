import os
import sys
import streamlit as st
import pandas as pd
from datetime import datetime

# Pengaturan Path untuk Folder Source (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

# ====== Cek Baris Import ======
from utils import load_model_artifacts, initial_text_clean, total_clean_pipeline

# ============================
# NORMALISASI FRASA (SINKRON)
# ============================
def normalize_phrases(text):
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
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# ============================
# KONFIGURASI HALAMAN UTAMA
# ============================
st.set_page_config(page_title="Indonesian Sentiment AI Analyst", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        [data-testid="stVerticalBlock"] > div:first-child > [data-testid="stHorizontalBlock"] {
            align-items: center;
            margin-bottom: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 15px;
            border-left: 8px solid #1f77b4;
        }
        .main-title { font-size: 38px; color: #1f77b4; font-weight: 800; text-align: left; margin-bottom: 5px; letter-spacing: 1px; }
        .sub-header { font-size: 15px; color: #444; text-align: left; font-style: italic; margin-bottom: 0px; }
        .stTextArea textarea { border-radius: 12px; border: 2px solid #1f77b4; font-size: 15px; background-color: #fff; }
        .stTextArea label { font-size: 16px; font-weight: 700; color: #1f77b4; }
        .stButton button { border-radius: 10px; font-weight: 700; transition: all 0.3s ease; }
        .stButton button:hover { transform: scale(1.02); }
        div[data-testid="stMetricValue"] { font-size: 26px; font-weight: bold; }
        div[data-testid="stMetricLabel"] { font-size: 14px; color: #333; }
        .sentiment-card { padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; background-color: #f8f9fa; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_models():
    return load_model_artifacts('models')

vectorizer, model = init_models()

# ============================
# KAMUS DATA 
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

DAFTAR_BLACKLIST = {
    "slot", "sl0t", "sl*t", "gacor", "g@cor", "judi", "judol",
    "bandar", "casino", "toto", "bet", "maxwin", "max win",
    "rtp", "rtp tinggi", "deposit", "wd", "wd cepat",
    "link", "alternatif", "iklan", "promosi", "paid-promote",
    "follow", "dm", "hubungi", "hubungi-wa",
    "dm_admin", "dm_slot", "dm_deposit", "dm_link", "dm_daftar", "dm_untuk", "dm_wa"
}

# ============================
# INTERPRETASI POLA & EMOSI
# ============================
NEGASI = {"tidak", "gak", "ga", "nggak", "tak", "tdk"}

KATA_POSITIF_PETUNJUK = {
    "bagus", "keren", "mantap", "ramah", "terjangkau", "puas",
    "suka", "senang", "nyaman", "aman", "cepat", "bersih",
    "indah", "rekomendasi", "terbaik", "hebat", "murah",
    "membantu", "untung", "lezat", "enak", "top", "mantul",
    "worth", "recommended", "memuaskan", "lebih_baik", "lancar_banget", "lebih_cepat"
}

KATA_NEGATIF_PETUNJUK = {
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
# FUNGSI DETEKSI LOGIKA
# ============================
def detect_negation_window(words):
    for i, w in enumerate(words):
        if w in NEGASI:
            window = words[i+1 : i+4]
            if any(next_word in KATA_POSITIF_PETUNJUK for next_word in window):
                return True
    return False

def detect_sarcasm_raw(raw_text):
    text_lower = raw_text.lower()
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
# STATE MANAGEMENT
# ============================
if 'hasil_analisis' not in st.session_state:
    st.session_state.teks_asli = None
    st.session_state.hasil_analisis = None
    st.session_state.sentiment_label = None
    st.session_state.prediction_label = None

# ============================
# UI HEADER
# ============================
header_col1, header_col2 = st.columns([1.8, 1.0])
with header_col1:
    st.markdown('<div class="main-title">INDONESIAN SENTIMENT AI ANALYST</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistem End-to-End Analisis Sentimen Multi-Kelas Bahasa Indonesia Berbasis Machine Learning</div>', unsafe_allow_html=True)

with header_col2:
    IMAGE_URL = "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=600&q=80"
    st.image(IMAGE_URL, use_container_width=True)

with st.expander("Catatan Transparansi AI & Keterbatasan Model"):
    st.write("""
    Sistem ini menggunakan gabungan filter aturan kata (*Strict Rule-Based*) dan *Machine Learning* (TF-IDF + Logistic Regression). 
    Alur pemrosesan dirancang secara *Defensive* untuk memastikan keluhan atau feedback negatif terlacak dengan akurat.
    """)

st.markdown("<br>", unsafe_allow_html=True)
col_space1, col_main, col_space2 = st.columns([1, 3, 1])

# ============================
# INPUT USER
# ============================
with col_main:
    user_input = st.text_area("✍️ Masukkan Teks Kalimat / Opini Publik Bahasa Indonesia:", height=130, placeholder="Contoh: Tempatnya bagus bgt, pelayanannya ramah...")
    analyze_button = st.button("🚀 Analisis Sentimen Sekarang", type="primary", use_container_width=True)

# ============================
# LOGIKA SINKRONISASI PIPELINE ANALISIS
# ============================
if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu!")
    else:
        st.session_state.teks_asli = user_input

        # 1. BARIKADE SPAM LINK
        if any(link in user_input.lower() for link in ['http', 'www', '.com', 'wa.me']):
            sentiment_val, predict_val = "Tidak Relevan", "Spam-Detected"
            cleaned_text = "(Teks diblokir karena mengandung tautan spam)"
        
        # 2. BARIKADE SARKASME MENTAH
        elif detect_sarcasm_raw(user_input):
            sentiment_val, predict_val = "Negatif", "Rule-Based Sarcasm Overrule"
            cleaned_text = normalize_phrases(total_clean_pipeline(user_input, SLANG_DICT))

        else:
            # CLEANING & NORMALISASI FRASA SINKRON
            cleaned_text = total_clean_pipeline(user_input, SLANG_DICT)
            cleaned_text = normalize_phrases(cleaned_text)
            
            kata_per_kata = [w for w in cleaned_text.split() if w.strip()]
            kata_set = set(kata_per_kata)

            # 3. BARIKADE TEKS KOSONG
            if not kata_per_kata:
                sentiment_val, predict_val = "Netral", "Rule-Based Empty Safe Filter"
                cleaned_text = "(Teks tidak mengandung kata bermakna)"

            # 4. BARIKADE BLACKLIST
            elif kata_set & DAFTAR_BLACKLIST:
                sentiment_val, predict_val = "Tidak Relevan", "Noise-Detected"

            # 5. BARIKADE WINDOW NEGATION
            elif detect_negation_window(kata_per_kata):
                sentiment_val, predict_val = "Negatif", "Rule-Based Window Negation"

            # JALUR EVALUASI NORMAL & KALIMAT CAMPURAN
            else:
                punya_kata_positif = bool(kata_set & KATA_POSITIF_PETUNJUK)
                punya_kata_negatif = bool(kata_set & KATA_NEGATIF_PETUNJUK)
                punya_keluhan = bool(kata_set & KELUHAN_KONTEKS)

                is_contrastive_positive = handle_contrastive_sentences(user_input)

                if (punya_kata_negatif or punya_keluhan) and is_contrastive_positive:
                    # Loloskan ke blok Machine Learning, biarkan model menentukan keputusan akhir
                    punya_kata_negatif = False
                    punya_keluhan = False
                    punya_kata_positif = True 

                # Cek Aturan Guard Ketat
                if punya_kata_negatif:
                    sentiment_val, predict_val = "Negatif", "Rule-Based Strict Negative Guard"
                elif punya_keluhan:
                    sentiment_val, predict_val = "Negatif", "Rule-Based Complaint Context Guard"
                
                # Kalimat pendek netral
                elif len(kata_per_kata) <= 2 and not (punya_kata_positif or punya_kata_negatif):
                    sentiment_val, predict_val = "Netral", "Rule-Based Short Text Filter"

                # Positif murni satu arah
                elif punya_kata_positif and not punya_kata_negatif:
                    sentiment_val, predict_val = "Positif", "Rule-Based Pure Positive"

                # Tidak memiliki indikator emosi
                elif not punya_kata_positif and not punya_kata_negatif:
                    sentiment_val, predict_val = "Netral", "Rule-Based Factual Filter"

                # 6. JALUR MODEL MACHINE LEARNING (CAMPURAN MURNI)
                else:
                    try:
                        text_tfidf = vectorizer.transform([cleaned_text])
                        pred_str = str(model.predict(text_tfidf)[0]).strip().lower()

                        if pred_str in ['positive', 'positif']:
                            sentiment_val = "Positif"
                        elif pred_str in ['negative', 'negatif']:
                            sentiment_val = "Negatif"
                        else:
                            sentiment_val = "Netral"

                        predict_val = "Logistic Regression Model (Mixed Text Decision)"

                    except Exception as e:
                        sentiment_val, predict_val = "Error", f"Sistem Gagal Memproses: {str(e)}"

        st.session_state.hasil_analisis = cleaned_text
        st.session_state.sentiment_label = sentiment_val
        st.session_state.prediction_label = predict_val

# ============================
# OUTPUT DIAGNOSIS
# ============================
if st.session_state.hasil_analisis:
    st.markdown("---")
    st.markdown("<h4 style='text-align: center; margin-bottom: 20px; color:#333;'>📊 Hasil Diagnosis Sentimen</h4>", unsafe_allow_html=True)
    
    col_res1, col_res2 = st.columns([1.8, 1.2])
    with col_res1:
        st.markdown(f"""
            <div class="sentiment-card">
                <p style="margin-bottom:8px; font-weight:bold; color:#1f77b4; font-size:14px;">📝 Teks Bersih (Pasca Preprocessing):</p>
                <p style="font-size:15px; font-style:italic; color:#222; margin-bottom:12px;">"{st.session_state.hasil_analisis}"</p>
                <hr style="margin:10px 0; border:0; border-top:1px solid #ddd;">
                <p style="font-size:11px; color:#666; margin:0;">Mekanisme Pendeteksi: {st.session_state.prediction_label}</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_res2:
        if st.session_state.sentiment_label == "Positif":
            st.metric(label="Status Hasil Sentimen", value="🟢 POSITIF", delta="Komentar Bernilai Baik/Pujian")
        elif st.session_state.sentiment_label == "Negatif":
            st.metric(label="Status Hasil Sentimen", value="🔴 NEGATIF", delta="Kritik / Keluhan Butuh Tindakan", delta_color="inverse")
        elif st.session_state.sentiment_label == "Netral":
            st.metric(label="Status Hasil Sentimen", value="⚪ NETRAL", delta="Pernyataan Fakta / Informasi Datar", delta_color="off")
        elif st.session_state.sentiment_label == "Tidak Relevan":
            st.metric(label="Status Hasil Sentimen", value="⚠️ DITOLAK", delta=st.session_state.prediction_label, delta_color="off")
        else:
            st.metric(label="Status Hasil Sentimen", value="❌ ERROR", delta=st.session_state.prediction_label, delta_color="off")

    # ============================
    # HUMAN-IN-THE-LOOP FEEDBACK
    # ============================
    st.markdown("---")
    st.markdown("#### 📢 Bantu Kami Evaluasi Model")
    st.write("Apakah hasil sentimen di atas sudah akurat dengan maksud kalimatmu?")
    
    col_fb1, col_fb2, col_fb_empty = st.columns([1, 1, 4])
    
    with col_fb1:
        if st.button("👍 Ya, Tepat", use_container_width=True):
            st.success("Terima kasih! Penilaianmu membantu performa analitik.")
            
    with col_fb2:
        if st.button("👎 Tidak Tepat", use_container_width=True):
            try:
                os.makedirs('data', exist_ok=True)
                csv_feedback_path = 'data/data_koreksi.csv'
                
                # SINKRONISASI LABEL: Gunakan 'positive'/'negative' standar dataset (huruf kecil)
                label_koreksi = "positive" if st.session_state.sentiment_label.lower() in ["negatif", "negative"] else "negative"
                
                df_koreksi = pd.DataFrame([{
                    "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "review_text": st.session_state.teks_asli,                 # Menyelaraskan nama kolom dengan main_nlp
                    "sentiment": label_koreksi,                              # <--- DIUBAH: Menyimpan label manusia yang BENAR!
                    "teks_bersih": st.session_state.hasil_analisis,
                    "prediksi_sistem_sebelumnya": st.session_state.sentiment_label, 
                    "sumber_prediksi": "human_feedback_correction"
                }])
                
                if not os.path.exists(csv_feedback_path):
                    df_koreksi.to_csv(csv_feedback_path, index=False, sep=';', encoding='utf-8')
                else:
                    df_koreksi.to_csv(csv_feedback_path, mode='a', header=False, index=False, sep=';', encoding='utf-8')
                    
                st.success(f"Log disimpan sebagai target label koreksi '{label_koreksi.upper()}'. Data ini akan dievaluasi pada retraining berikutnya.")
            except Exception as e:
                st.error(f"Gagal menyimpan feedback: {e}")