# ============================================================
# prediction.py
# Berisi seluruh logic untuk menu Prediksi Kategori Berita.
# Model final yang dipakai adalah ensemble soft voting dari
# empat model ANN Improvement: model_2 (baseline TextCNN),
# model_2b (num_filters=128), model_2c (dropout=0.3), dan
# model_3 (TextCNN dengan pretrained embedding Word2Vec).
# ============================================================
# Catatan penting: fungsi clean_text di file ini SENGAJA disamakan
# persis dengan fungsi clean_text pada tahap Feature Engineering di
# notebook utama, termasuk stopword removal Sastrawi. Kalau langkah
# stopword removal ini terlewat, distribusi kata pada teks baru akan
# berbeda dari distribusi yang dipelajari model saat training,
# sehingga bisa menurunkan akurasi prediksi tanpa disadari.

import re
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

MODEL_DIR = "saved_model"
MAX_LENGTH = 15
DEFAULT_THRESHOLD = 0.60


# ============================================================
# Load Artifacts (di-cache supaya model cuma dimuat sekali)
# ============================================================
@st.cache_resource(show_spinner="Memuat model ensemble dan artifact pendukung...")
def load_artifacts():
    # Import TensorFlow di sini, bukan di top-level file, supaya menu EDA tidak ikut menanggung memory footprint TensorFlow kalau user tidak pernah membuka menu Prediksi
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model_2 = load_model(f"{MODEL_DIR}/model_2_baseline.keras")
    model_2b = load_model(f"{MODEL_DIR}/model_2b_numfilters128.keras")
    model_2c = load_model(f"{MODEL_DIR}/model_2c_dropout03.keras")
    model_3 = load_model(f"{MODEL_DIR}/model_3_word2vec.keras")

    with open(f"{MODEL_DIR}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    models = [model_2, model_2b, model_2c, model_3]
    return models, tokenizer, label_encoder


@st.cache_resource
def get_stopword_remover():
    factory = StopWordRemoverFactory()
    return factory.create_stop_word_remover()


def clean_text(text, stopword_remover):
    """
    Fungsi ini harus identik dengan clean_text di tahap Feature
    Engineering pada notebook utama (cell 5.1), supaya teks baru
    diproses dengan cara yang persis sama dengan data training.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = stopword_remover.remove(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_category_batch(texts, models, tokenizer, label_encoder, max_length, stopword_remover):
    # Import library pad_sequences
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    texts_clean = [clean_text(t, stopword_remover) for t in texts]

    seq = tokenizer.texts_to_sequences(texts_clean)
    pad = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")

    proba_list = [model.predict(pad, verbose=0) for model in models]
    proba_avg = np.mean(proba_list, axis=0)

    pred_idx = np.argmax(proba_avg, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)
    confidences = np.max(proba_avg, axis=1)

    return texts_clean, pred_labels, confidences, proba_avg


# ============================================================
# Entry Point Halaman Prediksi
# ============================================================
def run():
    st.title("Prediksi Kategori Berita")
    st.markdown(
        "Masukkan judul berita baru dalam Bahasa Indonesia, model akan memprediksi "
        "kategorinya ke salah satu dari lima kategori berikut: **finance, health, "
        "oto, sport, travel**."
    )

    models, tokenizer, label_encoder = load_artifacts()
    stopword_remover = get_stopword_remover()

    threshold = st.slider(
        "Ambang batas confidence untuk flag Perlu Review Manual",
        min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.05,
        help=(
            "Kalau confidence prediksi model di bawah ambang batas ini, judul akan "
            "ditandai Perlu Review Manual, jadi tetap perlu dicek manusia sebelum "
            "dipakai. Berguna untuk kasus judul yang ambigu atau memuat topik lintas kategori."
        )
    )

    mode = st.radio("Mode Input", options=["Satu Judul", "Banyak Judul (Batch)"], horizontal=True)

    # ------------------------------------------------------------------
    # Mode Satu Judul
    # ------------------------------------------------------------------
    if mode == "Satu Judul":
        judul_input = st.text_input(
            "Judul Berita",
            placeholder="contoh: harga emas hari ini naik tipis di tengah ketidakpastian ekonomi global"
        )
        tombol = st.button("Prediksi Kategori", type="primary")

        if tombol:
            if not judul_input.strip():
                st.warning("Judul berita tidak boleh kosong.")
            else:
                judul_clean, labels, confs, proba_matrix = predict_category_batch(
                    [judul_input], models, tokenizer, label_encoder, MAX_LENGTH, stopword_remover
                )

                kategori_prediksi = labels[0]
                confidence = confs[0]

                kol1, kol2 = st.columns([1, 1])
                with kol1:
                    st.metric("Kategori Prediksi", kategori_prediksi.upper())
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    if confidence < threshold:
                        st.warning("Perlu Review Manual, confidence di bawah ambang batas.")
                    else:
                        st.success("Confidence di atas ambang batas, prediksi cukup meyakinkan.")

                with kol2:
                    df_proba = pd.DataFrame({
                        "Kategori": label_encoder.classes_,
                        "Probabilitas": proba_matrix[0]
                    }).sort_values("Probabilitas", ascending=False)
                    st.bar_chart(df_proba.set_index("Kategori"))

                with st.expander("Lihat detail hasil text cleaning"):
                    st.write("**Judul asli:**", judul_input)
                    st.write("**Judul setelah cleaning:**", judul_clean[0])

    # ------------------------------------------------------------------
    # Mode Batch
    # ------------------------------------------------------------------
    else:
        st.caption("Masukkan satu judul berita per baris.")
        teks_batch = st.text_area(
            "Daftar Judul Berita",
            height=180,
            placeholder=(
                "harga emas hari ini naik tipis di tengah ketidakpastian ekonomi global\n"
                "menteri kesehatan minta masyarakat waspada gejala flu singapura\n"
                "test drive mobil listrik terbaru dengan jarak tempuh 500 km"
            )
        )
        tombol_batch = st.button("Prediksi Semua Judul", type="primary")

        if tombol_batch:
            daftar_judul = [baris.strip() for baris in teks_batch.split("\n") if baris.strip()]
            if not daftar_judul:
                st.warning("Belum ada judul yang dimasukkan.")
            else:
                _, labels, confs, proba_matrix = predict_category_batch(
                    daftar_judul, models, tokenizer, label_encoder, MAX_LENGTH, stopword_remover
                )

                hasil = pd.DataFrame({
                    "Judul_Berita": daftar_judul,
                    "Prediksi_Kategori": labels,
                    "Confidence": confs
                })
                df_proba_all = pd.DataFrame(proba_matrix, columns=label_encoder.classes_)
                hasil = pd.concat([hasil, df_proba_all], axis=1)
                hasil["Perlu_Review_Manual"] = hasil["Confidence"] < threshold

                st.dataframe(hasil, use_container_width=True)

                csv_hasil = hasil.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Hasil Prediksi (CSV)",
                    data=csv_hasil,
                    file_name="hasil_prediksi_kategori_berita.csv",
                    mime="text/csv"
                )

                jumlah_review = int(hasil["Perlu_Review_Manual"].sum())
                if jumlah_review > 0:
                    st.warning(f"{jumlah_review} dari {len(hasil)} judul ditandai perlu review manual.")
