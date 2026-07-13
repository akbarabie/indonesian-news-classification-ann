# ============================================================
# app.py
# Entry point aplikasi Streamlit untuk project klasifikasi
# otomatis kategori berita Indonesia menggunakan ANN
# ============================================================
# File ini cuma bertugas mengatur konfigurasi halaman dan
# navigasi antar menu. Seluruh logic EDA dan Prediksi
# sengaja dipisah ke file eda.py dan prediction.py masing
# masing supaya kode tetap modular dan gampang dirawat,
# bukan ditumpuk semua di satu file besar.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
import streamlit as st
import eda
import prediction

# konfigurasi halaman harus dipanggil paling awal sebelum
# perintah streamlit lainnya, kalau tidak akan error
st.set_page_config(
    page_title="Klasifikasi Kategori Berita Indonesia",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Sidebar Navigasi
# ============================================================
with st.sidebar:
    st.title("📰 Navigasi")
    st.markdown(
        "Aplikasi berbasis kecerdasan buatan (AI) yang dirancang untuk mengklasifikasikan judul berita Indonesia secara otomatis menggunakan Artificial Neural Network (ANN). Aplikasi ini mengintegrasikan teknik Natural Language Processing (NLP) untuk memproses teks tidak terstruktur dan menghasilkan prediksi kategori berita yang akurat serta relevan."
    )

    menu = st.radio(
        "Pilih Menu",
        options=["Exploratory Data Analysis", "Prediksi Kategori Berita"],
        index=0
    )

    st.markdown("---")
    st.caption(
        "Model final yang dipakai adalah ensemble soft voting dari "
        "empat model ANN Improvement (TextCNN dengan variasi hyperparameter "
        "dan TextCNN dengan pretrained embedding Word2Vec)."
    )
    st.caption("Dibuat oleh Muhammad Akbar Suharbi" 
               "\nData & AI Professional")

# ============================================================
# Routing antar halaman berdasarkan pilihan menu
# ============================================================
if menu == "Exploratory Data Analysis":
    eda.run()
else:
    prediction.run()
