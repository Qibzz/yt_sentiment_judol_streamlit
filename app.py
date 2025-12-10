import streamlit as st
import pickle

# 1. Load Model & TF-IDF
model = pickle.load(open("model_nb.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# 2. UI Title
st.title("🎰 Sentiment Analysis Komentar YouTube tentang Judol (Naive Bayes)")
st.write("Aplikasi ini memprediksi apakah komentar mendukung atau menolak judi online.")

# 3. Input User
text = st.text_area("Masukkan komentar YouTube:")

# 4. Predict Button
if st.button("Prediksi Sentimen"):
    if text.strip() == "":
        st.warning("Kolom komentar masih kosong!")
    else:
        # TF-IDF transform
        tfidf_input = vectorizer.transform([text])
        
        # Prediction
        pred = model.predict(tfidf_input)[0]

        # Labeling output
        if pred == 1:
            st.success("🟢 POSITIVE — Mendukung Judol")
        else:
            st.error("🔴 NEGATIVE — Menolak Judol")

        # Show raw number (opsional)
        st.write("Prediksi model (0=Negative, 1=Positive):", int(pred))
