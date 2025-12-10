import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import io
import base64

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(
    page_title="Sentiment Analysis Judol",
    layout="wide",
    page_icon="🎰"
)

# ============================
# Load Model & TF-IDF (cache)
# ============================
@st.cache_resource
def load_model():
    model = pickle.load(open("model_nb.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ============================
# Load Dataset Labeled (cache)
# ============================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("data_labeled_final.csv")   # ← file kamu
        return df
    except FileNotFoundError:
        return None

df = load_dataset()

# ============================
# Fungsi Prediksi Sentimen
# ============================
def prediksi_sentimen(teks: str) -> int:
    tfidf_input = vectorizer.transform([teks])
    hasil = model.predict(tfidf_input)[0]
    return int(hasil)  # 0 = negatif, 1 = positif

# ============================
# Sidebar Navigasi
# ============================
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Home", "Dataset & Statistik", "Wordcloud", "Batch Prediksi", "About"]
)

# ============================
# 🏠 MENU HOME
# ============================
if menu == "Home":
    st.title("🎰 Sentiment Analysis Komentar YouTube tentang Judol (Naive Bayes)")
    st.write("Aplikasi ini memprediksi apakah komentar mendukung atau menolak judi online.")

    teks = st.text_area("Masukkan komentar YouTube:")

    if st.button("Prediksi Sentimen"):
        if teks.strip() == "":
            st.warning("Isi komentar dahulu yaa!")
        else:
            hasil = prediksi_sentimen(teks)

            if hasil == 1:
                st.success("🟢 **POSITIVE — Mendukung Judol**")
            else:
                st.error("🔴 **NEGATIVE — Menolak Judol**")

            st.write(f"Prediksi model (0=Negative, 1=Positive): **{hasil}**")

            # Simpan untuk grafik
            st.session_state["last_prediction"] = hasil
            st.session_state["last_text"] = teks

# ============================
# 📊 MENU DATASET & STATISTIK
# ============================
elif menu == "Dataset & Statistik":
    st.title("📊 Dataset & Statistik Sentimen")

    if df is None:
        st.warning("File `data_labeled_final.csv` tidak ditemukan di repo.")
    else:
        st.subheader("📁 Informasi Dataset")
        st.write(f"Jumlah total data: **{len(df)}**")

        if "label" in df.columns:
            counts = df["label"].value_counts().sort_index()
            neg = int(counts.get(0, 0))
            pos = int(counts.get(1, 0))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jumlah Negative (0)", neg)
            with col2:
                st.metric("Jumlah Positive (1)", pos)

            # Bar chart
            fig, ax = plt.subplots()
            ax.bar(["Negative (0)", "Positive (1)"], [neg, pos], color=["red", "green"])
            ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Label Sentimen")
            st.pyplot(fig)

        with st.expander("Lihat 10 data teratas"):
            st.dataframe(df.head(10))


# ============================
# ☁️ MENU WORDCLOUD
# ============================
elif menu == "Wordcloud":
    st.title("☁️ Wordcloud Sentimen")

    if df is None:
        st.warning("Upload dulu file `data_labeled_final.csv` ke GitHub.")
    else:
        if "label" not in df.columns or "stemmed" not in df.columns:
            st.error("Dataset harus memiliki kolom `label` & `stemmed`.")
        else:
            col1, col2 = st.columns(2)

            # Wordcloud Negatif
            with col1:
                st.subheader("🔴 Negative (label = 0)")
                teks_neg = " ".join(df[df["label"] == 0]["stemmed"].astype(str))
                if teks_neg.strip() == "":
                    st.info("Tidak ada data negatif.")
                else:
                    wc_neg = WordCloud(width=600, height=400, background_color="white").generate(teks_neg)
                    fig_neg, ax_neg = plt.subplots()
                    ax_neg.imshow(wc_neg, interpolation="bilinear")
                    ax_neg.axis("off")
                    st.pyplot(fig_neg)

            # Wordcloud Positif
            with col2:
                st.subheader("🟢 Positive (label = 1)")
                teks_pos = " ".join(df[df["label"] == 1]["stemmed"].astype(str))
                if teks_pos.strip() == "":
                    st.info("Tidak ada data positif.")
                else:
                    wc_pos = WordCloud(width=600, height=400, background_color="white").generate(teks_pos)
                    fig_pos, ax_pos = plt.subplots()
                    ax_pos.imshow(wc_pos, interpolation="bilinear")
                    ax_pos.axis("off")
                    st.pyplot(fig_pos)


# ============================
# 📥 MENU BATCH PREDIKSI
# ============================
elif menu == "Batch Prediksi":
    st.title("📥 Batch Prediksi & Download")

    st.write(
        "Upload file CSV berisi komentar, sistem akan memprediksi semuanya dan dapat didownload."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)

        st.write("📂 Contoh data:")
        st.dataframe(df_input.head())

        # cari kolom teks
        text_col = None
        if "stemmed" in df_input.columns:
            text_col = "stemmed"
        elif "text" in df_input.columns:
            text_col = "text"
        else:
            # pilih kolom teks dari user
            text_candidates = df_input.select_dtypes(include=["object"]).columns.tolist()
            text_col = st.selectbox("Pilih kolom teks:", text_candidates)

        if st.button("Prediksi Batch"):
            df_pred = df_input.copy()
            df_pred[text_col] = df_pred[text_col].astype(str).fillna("")

            tfidf_batch = vectorizer.transform(df_pred[text_col])
            preds = model.predict(tfidf_batch)

            df_pred["pred_label"] = preds
            df_pred["pred_kategori"] = df_pred["pred_label"].map(
                {0: "Negative (menolak judol)", 1: "Positive (mendukung judol)"}
            )

            st.success("Prediksi selesai!")
            st.dataframe(df_pred.head())

            # Download CSV
            csv_buffer = io.StringIO()
            df_pred.to_csv(csv_buffer, index=False)
            b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv">📥 Download Hasil Prediksi CSV</a>'
            st.markdown(href, unsafe_allow_html=True)


# ============================
# 📘 MENU ABOUT
# ============================
elif menu == "About":
    st.title("📘 Tentang Penelitian")

    st.subheader("🎯 Tujuan Penelitian")
    st.write(
        """
        Penelitian ini bertujuan untuk melakukan **analisis sentimen** terhadap komentar 
        YouTube yang berkaitan dengan judi online (judol). Sistem klasifikasi dibangun 
        menggunakan algoritma **Naive Bayes** dan **TF-IDF Vectorization**.
        """
    )

    st.subheader("🧠 Metodologi")
    st.write(
        """
        - **Dataset**: Komentar YouTube hasil scraping YouTube API  
        - **Preprocessing**:
            - Case folding  
            - Cleaning (emoji, URL, simbol, angka)  
            - Stopword removal  
            - Stemming  
        - **Feature Extraction**: TF-IDF  
        - **Model**: Multinomial Naive Bayes  
        """
    )

    st.subheader("⚙️ Tools yang Digunakan")
    st.write(
        """
        - Python  
        - Scikit-learn  
        - Pandas  
        - Streamlit  
        - Matplotlib  
        - Wordcloud  
        - Google Colab  
        - GitHub  
        """
    )

    st.subheader("👩‍🎓 Informasi Mahasiswa")
    st.write(
        """
        - **Nama**: Queen Qiblattul Qur'aini  
        - **NIM**: 5230411367  
        - **Universitas**: Universitas Teknologi Yogyakarta (UTY)  
        - **Prodi**: Informatika  
        """
    )

    st.info("Saya masih belajar :)")
