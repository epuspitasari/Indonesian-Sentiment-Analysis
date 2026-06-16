# Proyek 2: Klasifikasi Sentimen Multi-Kelas pada Teks Komentar Bahasa Indonesia

## 1. Pendahuluan & Latar Belakang
Proyek ini mengotomatisasi klasifikasi sentimen teks (Positif, Negatif, Netral) menggunakan *Natural Language Processing* (NLP) dan *Machine Learning*. Pendekatan ini dirancang untuk mengubah opini massal tidak terstruktur menjadi wawasan terukur guna mendukung pengambilan keputusan strategis yang bebas dari bias ketidakseimbangan kelas.

---

## 2. Dataset & Arsitektur Pipeline
* **Sumber Data:** Korpus *SmSA* (IndoNLU) via Alvian (Kaggle).
* **Partisi:** Data Latih (10.933 baris), Data Validasi (1.260 baris), dan Data Uji (500 baris dengan *Masked Label*).
* **Alur:** `[Raw Data]` ➔ `[01_Preprocessing]` ➔ `[02_Feature Engineering & Modeling]` ➔ `[03_Evaluasi & Deployment]`.

---

## 3. Ringkasan Teknis (Notebook 1-3)

### [Notebook 1] Preprocessing & EDA
* **Fokus:** Pembersihan noise teks (URL, akun mention, hashtag), normalisasi kata slang/singkatan menggunakan kamus terpemetaan, dan audit duplikasi data.
* **Output:** Dataset bersih terstandarisasi yang siap ditransformasikan ke dalam bentuk vektor numerik.

### [Notebook 2] Feature Engineering & Pemodelan
* **Teknik Ekstraksi:** `TfidfVectorizer` dengan parameter Kombinasi N-Gram ($ngram\_range=(1,2)$) dan pembatasan $max\_features=5000$ fitur teratas.
* **Penanganan Data Timpang:** Eksperimen awal menunjukkan model *baseline* mengalami "kebutaan" terhadap kelas minoritas (Recall kelas `neutral` rendah, ~0.66). Intervensi dilakukan menggunakan algoritma **SMOTE** yang diisolasi murni hanya pada komponen data latih (`X_train`) guna mendongkrak metrik *Recall* tanpa memicu kebocoran data (*data leakage*).
* **Model Terbaik:** `LogisticRegression` dengan penyesuaian parameter `class_weight='balanced'` terpilih sebagai model juara (champion model) karena menghasilkan performa metrik yang paling stabil, seimbang di seluruh kelas, serta memiliki kecepatan inferensi yang tinggi.

### [Notebook 3] Evaluasi & Produksi (Hybrid Architecture)
* **Validasi Independen:** Performa diuji secara ketat pada data validasi independen (bukan data uji *masked*) dengan capaian **Akurasi 87%** dan **Macro F1-Score 0.83**.
* **Hybrid Guardrails:** Temuan simulasi menunjukkan model ML berbasis linear rentan salah memprediksi kalimat faktual netral (misalnya: "saya tidak mampir"). Untuk memitigasi celah ini, kami mengimplementasikan **Defensive Guard** (Rule-Based) pada API produksi sebagai penyaring awal sebelum teks diteruskan ke model *Machine Learning*.
* **Distribusi Prediksi Unseen Data:** Hasil uji coba prediksi akhir pada 500 ulasan tersembunyi (*masked test set*) menghasilkan distribusi: **56.20% Negatif, 30.40% Positif, dan 13.40% Netral**.  

### Simulasi Bisnis & Dampak Operasional (ROI)
Guna mengukur dampak riil model di industri (misal: tim Customer Experience dengan volume 10.000 ulasan/bulan), dilakukan analisis biaya menggunakan *Cost-Benefit Matrix*:
* **Proses Manual (Tanpa ML):** Biaya peninjauan manual Rp2.000/teks. Total biaya operasional = **Rp20.000.000/bulan**.
* **Proses Otomatis (Hybrid ML 87% Accuracy):**
  * Biaya komputasi/API untuk 10.000 ulasan = Rp200.000.
  * Biaya *Human-in-the-loop* untuk meninjau ulang 13% error/miskasifikasi (1.300 teks x Rp2.000) = Rp2.600.000.
  * Total Biaya Baru = **Rp2.800.000/bulan**.
* **Dampak Finansial:** Menghemat biaya operasional sebesar **86% (Efisiensi sebesar Rp17.200.000/bulan)** serta memangkas waktu pemrosesan dari hitungan hari menjadi *real-time* detik.

---

## 4. Deployment & Produksi
Artefak model (`tfidf_vectorizer.pkl` dan `sentiment_model.pkl`) diintegrasikan ke dalam ekosistem siap pakai:
1. **Backend (FastAPI):** Menyediakan layanan API *real-time* yang mengombinasikan *Strict Guardrails* (Rule-Based) dan model *Machine Learning* untuk menjamin konsistensi dan objektivitas prediksi.
2. **Frontend (Streamlit):** Dasbor interaktif untuk riset pasar praktis yang dilengkapi dengan fitur *Human-in-the-loop* (fasilitas bagi pengguna untuk melakukan pelabelan ulang jika prediksi model kurang akurat).
3. **Mekanisme Retraining:** Setiap koreksi *feedback* dari antarmuka Streamlit akan disimpan secara otomatis ke dalam berkas `data/data_koreksi.csv` menggunakan separator titik koma (`;`). Target label distandarisasi menjadi huruf kecil seluruhnya (`positive`, `negative`, `neutral`) untuk memastikan data transisi tersebut dapat langsung dibaca sempurna oleh skrip otomatisasi `main_nlp.py` saat siklus *retraining* berkala dijalankan.

---

## 5. Rekomendasi Pengembangan (Future Work)
1. **Peningkatan Arsitektur Kontekstual:** Mengeksplorasi model berbasis *Transformer* (seperti **IndoBERT**) untuk menangani struktur kalimat dengan sarkasme tinggi, ambiguitas, dan negasi kompleks yang belum terakomodasi sempurna oleh model linear.
2. **Skalabilitas:** Mengoptimalkan sistem penyimpanan berbasis *caching* pada API backend untuk menangani volume *traffic* ulasan yang lebih masif secara *asynchronous*.

---
*Proyek ini merupakan bagian dari siklus pengembangan kompetensi Data Science di Pacmann.*