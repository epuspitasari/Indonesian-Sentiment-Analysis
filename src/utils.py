# utils.py
import os
import re
import joblib

def initial_text_clean(text):
    """
    Melakukan pembersihan teks tahap awal secara komprehensif.

    Proses yang dilakukan:
    1. Case folding (mengubah teks menjadi huruf kecil).
    2. Penghapusan URL atau tautan web (http/https/www).
    3. Penghapusan akun mention (@username) dan hashtag (#tag).
    4. Penghapusan karakter non-alfabet (angka, tanda baca, emoji) menjadi spasi tunggal.
    5. Pembersihan spasi berlebih (whitespaces) di awal, tengah, dan akhir teks.

    Parameters:
    ----------
    text : str
        Teks mentah inputan yang akan dibersihkan.

    Returns:
    -------
    str
        Teks yang telah dibersihkan dari noise awal dan siap dinormalisasi.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\S+|#\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def total_clean_pipeline(text, dictionary):
    """
    Pipeline preprocessing end-to-end yang menggabungkan pembersihan teks awal 
    dan normalisasi kata tidak baku (slang/singkatan) berdasarkan kamus terpetakan.
    Dilengkapi dengan proteksi string kosong untuk menjaga kestabilan inferensi.

    Parameters:
    ----------
    text : str
        Teks mentah inputan dari user atau dataset luar.
    dictionary : dict
        Kamus pemetaan kata gaul/singkatan ke kata baku (misal: {'yg': 'yang'}).

    Returns:
    -------
    str
        Teks akhir yang telah bersih, rapi, baku, dan siap diekstrak menjadi matriks TF-IDF.
    """
    text_cleaned = initial_text_clean(text)
    words = text_cleaned.split()
    
    # Kondisi Proteksi 1: Jika inputan kosong setelah dibersihkan regex, kembalikan string kosong penanda
    if len(words) == 0:
        return "teks kosong"
        
    normalized_words = [dictionary.get(word, word) for word in words]
    final_text = " ".join(normalized_words)
    
    # Kondisi Proteksi 2: Memastikan output akhir tidak berupa string kosong melompong
    if not final_text.strip():
        return "teks kosong"
        
    return final_text

def save_model_artifacts(model, filepath):
    """
    Menyimpan objek model hasil training atau tuning ke dalam format .pkl.
    Otomatis mendeteksi jika objek berupa GridSearchCV untuk mengambil best_estimator_.
    
    Parameters:
    -----------
    model : estimator object
        Model yang akan disimpan (bisa berupa model scikit-learn atau GridSearchCV).
    filepath : str
        Jalur lengkap lokasi penyimpanan file .pkl.
    """
    # Deteksi otomatis hasil GridSearchCV vs model biasa
    model_to_save = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
    
    # Pastikan folder tempat menyimpan tersedia
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    joblib.dump(model_to_save, filepath)
    print(f"=== [SUCCESS] Berhasil disimpan di: {filepath} ===")

def load_model_artifacts(model_dir='models'):
    """
    Fungsi utilitas pendukung deployment untuk memuat berkas biner (.pkl)
    TfidfVectorizer dan Model Klasifikasi secara aman dari berbagai hirarki folder.
    Mendukung auto-fallback path jika dipanggil dari folder utama maupun subfolder.

    Parameters:
    ----------
    model_dir : str, default 'models'
        Jalur direktori tempat file artifact .pkl disimpan.

    Returns:
    -------
    tuple (vectorizer, model) atau (None, None)
        Mengembalikan objek vectorizer dan model jika sukses, atau None jika gagal.
    """
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    
    # Deteksi otomatis jalur alternatif jika file tidak langsung ketemu 
    # (Sangat berguna saat perpindahan eksekusi antara Jupyter Notebook dan Streamlit/FastAPI)
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        # Jalur alternatif 1: langsung tembak folder 'models' di root
        if os.path.exists(os.path.join('models', 'tfidf_vectorizer.pkl')):
            vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
            model_path = os.path.join('models', 'sentiment_model.pkl')
        # Jalur alternatif 2: naik satu tingkat ke folder induk (untuk Jupyter Notebook)
        elif os.path.exists(os.path.join('..', 'models', 'tfidf_vectorizer.pkl')):
            vectorizer_path = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')
            model_path = os.path.join('..', 'models', 'sentiment_model.pkl')
        
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        print("=== [SUCCESS] Semua Model Artifact Berhasil Dimuat Lewat Utilitas! ===")
        return vectorizer, model
    except Exception as e:
        print(f"=== [ERROR] Gagal memuat artifact model: {str(e)} ===")
        return None, None