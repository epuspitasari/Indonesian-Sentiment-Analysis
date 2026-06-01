# Proyek 2: Klasifikasi Sentimen Multi-Kelas pada Teks Komentar Bahasa Indonesia

## 1. Pendahuluan & Latar Belakang
Proyek ini mengotomatisasi klasifikasi sentimen teks (Positif, Negatif, Netral) menggunakan *Natural Language Processing* (NLP) dan *Machine Learning*. Pendekatan ini dirancang untuk mengubah opini massal tidak terstruktur menjadi wawasan terukur guna mendukung pengambilan keputusan strategis.

---

## 2. Dataset & Arsitektur Pipeline
* **Sumber Data:** Korpus *SmSA* (IndoNLU) via Alvian (Kaggle).
* **Partisi:** Train (10.933 baris), Validation (1.260 baris), Test (500 baris dengan *Masked Label*).
* **Alur:** `[Raw Data]` ➔ `[01_Preprocessing]` ➔ `[02_Feature Engineering & Modeling]` ➔ `[03_Evaluasi & Deployment]`.

---

## 3. Ringkasan Teknis (Notebook 1-3)

### [Notebook 1] Preprocessing & EDA
* **Fokus:** Pembersihan noise (URL, mention, hashtag), normalisasi slang/singkatan, dan audit duplikat.
* **Output:** Dataset bersih yang siap diolah menjadi vektor numerik.

### [Notebook 2] Feature Engineering & Pemodelan
* **Teknik:** `TfidfVectorizer` ($ngram=(1,2)$, $max\_features=5000$).
* **Handling Imbalance:** Penggunaan **SMOTE** untuk menyeimbangkan kelas minoritas. Tanpa SMOTE, model cenderung memiliki bias tinggi (Recall kelas `neutral` rendah, ~0.66).
* **Model:** `LogisticRegression` (class_weight='balanced') terpilih sebagai model juara karena performa stabil dan kecepatan inferensi.

### [Notebook 3] Evaluasi & Produksi (Hybrid Architecture)
* **Validasi Independen:** Performa tervalidasi pada data validasi (bukan data uji *masked*): **Akurasi 87%** dan **Macro F1-Score 0.83**.
* **Hybrid Guardrails:** Temuan simulasi menunjukkan model ML murni rentan terhadap kalimat faktual netral ("saya tidak mampir"). Kami mengimplementasikan **Defensive Guard** (Rule-Based) pada API produksi untuk menetapkan label secara objektif sebelum diproses ML.
* **Distribusi Prediksi:** Uji coba pada 500 ulasan tersembunyi menghasilkan: **56.20% Negatif, 30.40% Positif, dan 13.40% Netral**.

---

## 4. Deployment & Produksi
Artefak model (`tfidf_vectorizer.pkl` dan `sentiment_model.pkl`) diintegrasikan ke dalam:
1.  **Backend (FastAPI):** Menggunakan *Strict Guardrails* untuk memproses sentimen secara *real-time*.
2.  **Frontend (Streamlit):** Dasbor interaktif untuk riset pasar praktis, dilengkapi fitur *Human-in-the-loop* (pelabelan ulang otomatis jika prediksi salah).
3.  Penyimpanan data koreksi feedback pada Streamlit disimpan ke dalam bentuk berkas data/data_koreksi.csv menggunakan separator ; dengan standarisasi target label huruf kecil (positive / negative) agar saat proses retraining otomatis terbaca sempurna oleh main_nlp.py.

---

## 5. Rekomendasi Pengembangan (Future Work)
1.  **Peningkatan Arsitektur:** Eksplorasi model *Deep Learning* (seperti **IndoBERT**) untuk menangani sarkasme dan negasi kompleks yang belum tertangani sempurna oleh model linear.
2.  **Skalabilitas:** Optimalisasi *caching* pada API untuk menangani volume ulasan yang lebih masif secara *asynchronous*.

---
*Proyek ini merupakan bagian dari siklus pengembangan kompetensi Data Science di Pacmann.*