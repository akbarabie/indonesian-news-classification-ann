# ============================================================
# eda.py
# Berisi seluruh logic untuk menu Exploratory Data Analysis
# Dipecah jadi dua sub bagian sesuai proses di notebook utama:
# 1. EDA Sebelum Feature Engineering (section 4 di notebook)
# 2. EDA Setelah Feature Engineering (section 5.7 di notebook)
# ============================================================
# Catatan penting: seluruh angka dan visualisasi di halaman ini
# dihitung ulang langsung dari dataset asli (bukan hasil hardcode),
# supaya kalau dataset berubah, analisis di aplikasi ini otomatis
# ikut menyesuaikan. Proses yang berat (cleaning teks, split data)
# di-cache pakai st.cache_data supaya cuma dihitung sekali per sesi,
# tidak diulang setiap kali user pindah tab atau widget.

import re
import pickle
from collections import Counter
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from PIL import Image

# ============================================================
# Konstanta, disamakan persis dengan yang dipakai di notebook utama
# ============================================================
DATASET_PATH = "detik_news_title.csv"
TOKENIZER_PATH = "saved_model/tokenizer.pkl"
KATEGORI_TERPILIH = ["finance", "health", "oto", "sport", "travel"]
VOCAB_SIZE = 10000
MAX_LENGTH = 15
SEED = 88

sns.set_style("whitegrid")


# ============================================================
# Data Loading dan Preprocessing (di-cache)
# ============================================================
@st.cache_data(show_spinner="Memuat dan menyiapkan dataset...")
def load_raw_data():
    """
    Replikasi persis proses Data Loading di notebook utama:
    load csv, hapus duplikat judul, convert tipe data date,
    lalu filter ke 5 kategori yang dipakai di project ini.
    Urutan step ini sengaja disamakan dengan notebook supaya
    angka yang keluar di aplikasi konsisten dengan yang sudah
    dianalisis dan dinarasikan di notebook.
    """
    df = pd.read_csv(DATASET_PATH)

    # hapus judul duplikat, judul yang sama persis dibuang supaya
    # tidak ada risiko data leakage antar split train val test
    df = df.drop_duplicates(subset="title", keep="first").reset_index(drop=True)

    # convert kolom date dari object ke datetime
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # filter dataset hanya ke 5 kategori yang jadi fokus project
    df = df[df["category"].isin(KATEGORI_TERPILIH)].reset_index(drop=True)

    # feature tambahan untuk analisis panjang judul
    df["title_word_count"] = df["title"].apply(lambda x: len(x.split()))
    df["title_char_count"] = df["title"].apply(len)

    return df


@st.cache_data(show_spinner="Membersihkan teks judul (text cleaning + stopword removal)...")
def build_title_clean(df):
    """
    Replikasi fungsi clean_text yang benar benar dipakai di tahap
    Feature Engineering notebook utama (cell 5.1), termasuk
    stopword removal Sastrawi. Ini BUKAN clean_text sederhana yang
    dipakai di EDA sebelum Feature Engineering untuk word frequency,
    dua duanya memang beda tujuan dan sengaja dipisah.
    """
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = stopword_remover.remove(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df = df.copy()
    df["title_clean"] = df["title"].apply(clean_text)
    df["clean_word_count"] = df["title_clean"].apply(lambda x: len(x.split()))
    return df


@st.cache_data(show_spinner="Membagi data menjadi train, validation, dan test set...")
def split_data(df_clean):
    """
    Replikasi persis proses train-val-test split di Feature Engineering,
    random_state dan proporsi split disamakan supaya split yang
    terbentuk di aplikasi ini identik dengan yang dipakai untuk training.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        df_clean["title_clean"], df_clean["category"],
        test_size=0.30, random_state=SEED, stratify=df_clean["category"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50, random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


@st.cache_resource(show_spinner="Memuat tokenizer...")
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def clean_text_simple(text):
    """
    Versi cleaning sederhana yang dipakai khusus untuk eksplorasi
    di EDA sebelum Feature Engineering (cell 4.3 dan 4.5), regex
    saja tanpa stopword removal. Sengaja didefinisikan di level
    module (bukan nested function) supaya bisa dipakai ulang di
    beberapa tempat dan tetap aman untuk di-cache oleh Streamlit.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner="Menyiapkan daftar stopword Bahasa Indonesia...")
def get_stopwords_id():
    stopword_factory = StopWordRemoverFactory()
    return set(stopword_factory.get_stop_words())


@st.cache_data(show_spinner="Menghitung word frequency per kategori...")
def compute_word_tokens(df):
    """
    Replikasi fungsi get_tokens di EDA sebelum Feature Engineering
    (cell 4.3), dipakai untuk word frequency dan wordcloud. Fungsi
    ini sengaja berbeda dari clean_text final karena tujuannya cuma
    eksplorasi, bukan input ke model. Hanya mengembalikan data
    (pandas Series), bukan function, karena st.cache_data mewajibkan
    hasil yang bisa di-pickle.
    """
    stopwords_id = get_stopwords_id()

    def get_tokens(text):
        cleaned = clean_text_simple(text)
        tokens = [w for w in cleaned.split() if w not in stopwords_id and len(w) > 2]
        return tokens

    tokens_per_row = df["title"].apply(get_tokens)
    return tokens_per_row


# ============================================================
# Sub Halaman 1 - EDA Sebelum Feature Engineering
# ============================================================
def render_before_fe(df):
    st.subheader("Exploratory Data Analysis (Before Feature Engineering)")
    st.caption(
        "Eksplorasi dilakukan pada data mentah yang sudah difilter ke 5 kategori, "
        "sebelum masuk ke tahap Feature Engineering (text cleaning, split data, "
        "dan tokenisasi)."
    )

    total = len(df)
    kolom1, kolom2, kolom3 = st.columns(3)
    kolom1.metric("Total Judul (5 Kategori)", f"{total:,}")
    kolom2.metric("Jumlah Kategori", "5")
    kolom3.metric("Rentang Waktu", "Jan - Jun 2020")

    # ------------------------------------------------------------------
    # 4.1 Distribusi Kategori
    # ------------------------------------------------------------------
    st.markdown("#### 4.1 Distribusi Kategori")
    category_counts = df["category"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=category_counts.index, y=category_counts.values, order=category_counts.index, ax=ax)
    for i, value in enumerate(category_counts.values):
        pct = value / total * 100
        ax.text(i, value + 200, f"{value}\n({pct:.1f}%)", ha="center", fontsize=9)
    ax.set_title("Distribusi Judul pada Setiap Kategori")
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Jumlah Judul")
    st.pyplot(fig)
    plt.close(fig)

    rasio = category_counts.max() / category_counts.min()
    st.info(
        f"Rasio antara kategori terbesar dan terkecil sekitar **{rasio:.1f} kali lipat** "
        f"({category_counts.idxmax()} dibanding {category_counts.idxmin()}). Class imbalance "
        "sebesar ini jadi alasan kenapa class_weight dipakai saat training di tahap "
        "Feature Engineering, supaya model tidak bias memprediksi kelas mayoritas saja."
    )

    # ------------------------------------------------------------------
    # 4.2 Distribusi Panjang Judul per Kategori
    # ------------------------------------------------------------------
    st.markdown("#### 4.2 Distribusi Panjang Judul per Kategori")

    kol1, kol2 = st.columns(2)
    with kol1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(df["title_word_count"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribusi Jumlah Kata pada Judul (Seluruh Kategori)")
        ax.set_xlabel("Jumlah Kata")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
        plt.close(fig)

    with kol2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="category", y="title_word_count", order=category_counts.index, ax=ax)
        ax.set_title("Jumlah Kata pada Judul untuk Setiap Kategori")
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah Kata")
        st.pyplot(fig)
        plt.close(fig)

    desc_word = df.groupby("category")["title_word_count"].describe().round(2)
    st.dataframe(desc_word, use_container_width=True)
    st.caption(
        f"Skewness: {df['title_word_count'].skew():.2f}, "
        f"Kurtosis: {df['title_word_count'].kurt():.2f}. "
        "Kedua nilai ini mendekati nol, artinya distribusi jumlah kata judul "
        "praktis simetris dan mendekati normal, tanpa ekor panjang yang perlu dikhawatirkan."
    )

    # ------------------------------------------------------------------
    # 4.3 Word Frequency per Kategori
    # ------------------------------------------------------------------
    st.markdown("#### 4.3 Word Frequency per Kategori")
    tokens_per_row = compute_word_tokens(df)
    df_tokens = df[["category"]].copy()
    df_tokens["tokens"] = tokens_per_row

    kategori_pilihan = st.selectbox(
        "Pilih kategori untuk melihat kata yang paling sering muncul",
        options=category_counts.index.tolist(),
        key="wordfreq_kategori"
    )
    tokens_flat = [t for tokens in df_tokens.loc[df_tokens["category"] == kategori_pilihan, "tokens"] for t in tokens]
    top_words = Counter(tokens_flat).most_common(15)
    df_top_words = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_top_words, x="Frekuensi", y="Kata", ax=ax, color="#4C72B0")
    ax.set_title(f"15 Kata Paling Sering Muncul - Kategori {kategori_pilihan}")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Kata **corona** menjadi kata teratas di hampir semua kategori (kecuali oto dimana "
        "mobil sedikit lebih sering), karena rentang data Januari - Juni 2020 bertepatan dengan "
        "awal pandemi. Di luar kata corona ini, tiap kategori tetap punya kosakata khas masing "
        "masing yang jelas berbeda, ini membuktikan hipotesis vocabulary separability yang jadi "
        "alasan pemilihan lima kategori ini di awal project."
    )

    # ------------------------------------------------------------------
    # 4.4 WordCloud per Kategori
    # ------------------------------------------------------------------
    st.markdown("#### 4.4 WordCloud per Kategori")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, cat in enumerate(category_counts.index):
        tokens_flat_cat = [t for tokens in df_tokens.loc[df_tokens["category"] == cat, "tokens"] for t in tokens]
        text_combined = " ".join(tokens_flat_cat)
        wc = WordCloud(width=600, height=400, background_color="white", colormap="viridis", random_state=SEED).generate(text_combined)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Category: {cat}")
        axes[i].axis("off")
    axes[5].axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4.5 N-gram Analysis
    # ------------------------------------------------------------------
    st.markdown("#### 4.5 N-gram Analysis (Bigram dan Trigram) per Kategori")

    stopwords_id = get_stopwords_id()
    df_cleaned_series = df["title"].apply(clean_text_simple)

    def get_top_ngrams(text_series, ngram_range, top_n=10):
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=list(stopwords_id))
        ngram_matrix = vectorizer.fit_transform(text_series)
        ngram_counts = ngram_matrix.sum(axis=0)
        ngram_freq = [(word, ngram_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        return sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:top_n]

    kategori_ngram = st.selectbox(
        "Pilih kategori untuk melihat bigram dan trigram teratas",
        options=category_counts.index.tolist(),
        key="ngram_kategori"
    )
    subset = df_cleaned_series[df["category"] == kategori_ngram]

    kol1, kol2 = st.columns(2)
    with kol1:
        st.markdown("**Top 10 Bigram**")
        df_bigram = pd.DataFrame(get_top_ngrams(subset, (2, 2)), columns=["Frasa", "Frekuensi"])
        st.dataframe(df_bigram, use_container_width=True, hide_index=True)
    with kol2:
        st.markdown("**Top 10 Trigram**")
        df_trigram = pd.DataFrame(get_top_ngrams(subset, (3, 3)), columns=["Frasa", "Frekuensi"])
        st.dataframe(df_trigram, use_container_width=True, hide_index=True)

    st.info(
        "Begitu dilihat dalam bentuk frasa dua atau tiga kata, karakter tiap kategori jadi jauh "
        "lebih tajam. Ini alasan kuat kenapa model ANN yang dipakai sebaiknya bisa menangkap "
        "urutan kata (misalnya lewat Conv1D pada TextCNN), bukan cuma bag-of-words biasa yang "
        "mengabaikan urutan kata."
    )

    # ------------------------------------------------------------------
    # 4.6 Tren Volume Berita per Waktu
    # ------------------------------------------------------------------
    st.markdown("#### 4.6 Tren Volume Berita per Waktu")
    df_trend = df.copy()
    df_trend["year_month"] = df_trend["date"].dt.to_period("M").astype(str)
    monthly_trend = df_trend.groupby(["year_month", "category"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=monthly_trend, x="year_month", y="count", hue="category", marker="o", ax=ax)
    ax.set_title("Volume Berita Bulanan Berdasarkan Kategori (Januari - Juni 2020)")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Judul")
    ax.legend(title="Kategori", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Health melonjak tajam dari sekitar 400-an judul di Februari jadi 1300-an di Maret, "
        "bertepatan dengan pengumuman kasus positif corona pertama di Indonesia awal Maret 2020. "
        "Penurunan tajam di Juni bukan tren asli, karena data cuma tersedia sampai tanggal 13."
    )

    # ------------------------------------------------------------------
    # 4.7 - 4.8 Ringkasan Pengecekan Anomali
    # ------------------------------------------------------------------
    st.markdown("#### 4.7 - 4.8 Ringkasan Pengecekan Anomali Judul")
    jumlah_pendek = int((df["title_word_count"] <= 2).sum())
    jumlah_mojibake = int(df["title"].str.contains("Â", regex=False).sum())

    kol1, kol2 = st.columns(2)
    kol1.metric("Judul Sangat Pendek (<= 2 kata)", jumlah_pendek)
    kol2.metric("Judul Mengandung Karakter Mojibake 'Â'", jumlah_mojibake)

    with st.expander("Lihat kesimpulan pengecekan anomali"):
        st.markdown(
            "Sudah dilakukan pengecekan menyeluruh terhadap judul yang sangat pendek, karakter "
            "tidak lazim, karakter mojibake, emoji, singkatan gaya chat, dan kandidat typo di "
            "tahap eksplorasi pada notebook utama. Kesimpulannya, tidak ada satupun temuan yang "
            "butuh penanganan khusus tambahan di luar pipeline cleaning yang sudah dirancang. "
            "Karakter mojibake 'Â' misalnya, tetap otomatis terbuang oleh fungsi `clean_text` "
            "karena bukan huruf a sampai z, sementara kata di sekitarnya tetap terpisah dengan benar."
        )


# ============================================================
# Sub Halaman 2 - EDA Setelah Feature Engineering
# ============================================================
def render_after_fe(df):
    st.subheader("Exploratory Data Analysis (After Feature Engineering)")
    st.caption(
        "Eksplorasi dilakukan setelah text cleaning (termasuk stopword removal) dan "
        "train-val-test split, untuk memverifikasi bahwa proses Feature Engineering "
        "berjalan sesuai rencana sebelum data masuk ke tahap training."
    )

    df_clean = build_title_clean(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # ------------------------------------------------------------------
    # 5.7.1 Verifikasi Distribusi Kategori pada Split
    # ------------------------------------------------------------------
    st.markdown("##### 5.7.1 Verifikasi Distribusi Kategori pada Train-Val-Test Split")

    proporsi = pd.DataFrame({
        "Train": (y_train.value_counts(normalize=True) * 100).round(2),
        "Validation": (y_val.value_counts(normalize=True) * 100).round(2),
        "Test": (y_test.value_counts(normalize=True) * 100).round(2),
    }).sort_values("Train", ascending=False)
    st.dataframe(proporsi, use_container_width=True)

    kol1, kol2, kol3 = st.columns(3)
    kol1.metric("Jumlah Data Train", f"{len(X_train):,}")
    kol2.metric("Jumlah Data Validation", f"{len(X_val):,}")
    kol3.metric("Jumlah Data Test", f"{len(X_test):,}")

    st.info(
        "Proporsi tiap kategori konsisten di ketiga split, ini bukti bahwa parameter `stratify` "
        "bekerja dengan benar. Konsistensi ini penting mengingat sport cuma 7.5% dari total data, "
        "kalau tidak distratify ada risiko sport kurang terwakili di validation atau test set."
    )

    # ------------------------------------------------------------------
    # 5.7.2 Distribusi Panjang Judul Setelah Text Cleaning
    # ------------------------------------------------------------------
    st.markdown("##### 5.7.2 Distribusi Panjang Judul Setelah Text Cleaning")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df_clean["clean_word_count"], bins=range(1, df_clean["clean_word_count"].max() + 2), kde=True, ax=ax)
    ax.axvline(MAX_LENGTH, color="red", linestyle="--", label=f"MAX_LENGTH = {MAX_LENGTH}")
    ax.set_title("Distribusi Jumlah Kata pada title_clean Setelah Text Cleaning")
    ax.set_xlabel("Jumlah Kata")
    ax.set_ylabel("Frekuensi")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    rata_rata = df_clean["clean_word_count"].mean()
    jumlah_terpotong = (df_clean["clean_word_count"] > MAX_LENGTH).sum()
    pct_terpotong = jumlah_terpotong / len(df_clean) * 100
    st.info(
        f"Rata rata panjang judul setelah cleaning adalah **{rata_rata:.2f} kata**, dengan "
        f"**{jumlah_terpotong} judul ({pct_terpotong:.4f}%)** yang melebihi MAX_LENGTH={MAX_LENGTH}. "
        "Ini membuktikan MAX_LENGTH=15 adalah pilihan yang aman, praktis tidak ada informasi "
        "judul yang hilang akibat proses truncating."
    )

    # ------------------------------------------------------------------
    # 5.7.3 Analisis Vocabulary Coverage dan OOV Rate
    # ------------------------------------------------------------------
    st.markdown("##### 5.7.3 Analisis Vocabulary Coverage dan OOV Rate")

    try:
        tokenizer = load_tokenizer()
        oov_index = tokenizer.word_index[tokenizer.oov_token]

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_val_seq = tokenizer.texts_to_sequences(X_val)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        def hitung_oov(seq_list):
            total_token = sum(len(seq) for seq in seq_list)
            total_oov = sum(seq.count(oov_index) for seq in seq_list)
            return total_oov, total_token, total_oov / total_token * 100

        _, _, rate_train = hitung_oov(X_train_seq)
        _, _, rate_val = hitung_oov(X_val_seq)
        _, _, rate_test = hitung_oov(X_test_seq)

        kol1, kol2, kol3 = st.columns(3)
        kol1.metric("OOV Rate Train", f"{rate_train:.2f}%")
        kol2.metric("OOV Rate Validation", f"{rate_val:.2f}%")
        kol3.metric("OOV Rate Test", f"{rate_test:.2f}%")

        def judul_ada_oov(seq_list):
            return [oov_index in seq for seq in seq_list]

        val_has_oov = pd.Series(judul_ada_oov(X_val_seq))
        test_has_oov = pd.Series(judul_ada_oov(X_test_seq))
        valtest_has_oov = pd.concat([val_has_oov, test_has_oov], ignore_index=True)
        valtest_category = pd.concat([y_val.reset_index(drop=True), y_test.reset_index(drop=True)], ignore_index=True)
        oov_per_kategori = (valtest_has_oov.groupby(valtest_category).mean() * 100).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=oov_per_kategori.index, y=oov_per_kategori.values, ax=ax, color="#DD8452")
        ax.set_title("Persentase Judul (Val+Test) dengan Minimal 1 Kata OOV per Kategori")
        ax.set_ylabel("Persentase (%)")
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Sport dan travel konsisten punya persentase judul mengandung OOV yang paling tinggi, "
            "sejalan dengan ukuran data training-nya yang paling sedikit. Semakin sedikit data "
            "training untuk satu kategori, semakin terbatas variasi kosakata yang sempat dipelajari "
            "tokenizer dari kategori tersebut."
        )
    except FileNotFoundError:
        st.warning(
            "File tokenizer.pkl belum ditemukan di folder saved_model. Bagian analisis OOV rate "
            "memerlukan tokenizer hasil training untuk bisa dihitung."
        )

    # ------------------------------------------------------------------
    # 5.7.4 WordCloud Long Tail Words per Kategori
    # ------------------------------------------------------------------
    st.markdown("##### 5.7.4 WordCloud Long Tail Words per Kategori")

    semua_kata_train = [tok for title in X_train for tok in title.split()]
    frekuensi_kata = Counter(semua_kata_train)
    sorted_words = [w for w, c in frekuensi_kata.most_common()]
    long_tail_words = set(sorted_words[VOCAB_SIZE - 1:])

    jumlah_freq_1 = sum(1 for w in long_tail_words if frekuensi_kata[w] == 1)
    kol1, kol2 = st.columns(2)
    kol1.metric("Jumlah Kata Long Tail", f"{len(long_tail_words):,}")
    kol2.metric("Hanya Muncul 1 Kali di Train", f"{jumlah_freq_1/len(long_tail_words)*100:.2f}%" if long_tail_words else "0%")

    df_train_cek = pd.DataFrame({"title_clean": X_train.values, "category": y_train.values})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, cat in enumerate(sorted(df_train_cek["category"].unique())):
        tokens_cat = [
            w for title in df_train_cek.loc[df_train_cek["category"] == cat, "title_clean"]
            for w in title.split() if w in long_tail_words
        ]
        text_combined = " ".join(tokens_cat)
        wc = WordCloud(width=600, height=400, background_color="white", colormap="magma", random_state=SEED).generate(text_combined)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Long Tail Words - Category: {cat}")
        axes[i].axis("off")
    axes[5].axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Long tail sport, oto, dan travel didominasi entitas nama spesifik seperti nama atlet, "
        "merek, dan destinasi. Sebaliknya, long tail finance didominasi kosakata umum, bukan "
        "entitas nama. Perbedaan sumber OOV ini penting karena solusi perbaikannya bisa berbeda, "
        "menaikkan VOCAB_SIZE mungkin cukup membantu finance, tapi kurang membantu kategori yang "
        "akarnya ada di named entity yang terus bertambah baru."
    )


# ============================================================
# Entry Point Halaman EDA
# ============================================================
def run():
    st.markdown(
        """
        <h1 style='text-align: center;'>
            Automatic Indonesian News Classification
        </h1>
        """,
        unsafe_allow_html=True
    )
    # Menampilkan cover gambar
    img = Image.open('image_news.png')
    st.image(img, caption="News Classification")
    st.markdown(
         """
        <p style='text-align:center; font-size:20px;'>
            "Halaman ini menampilkan eksplorasi data pada dataset **detik_news_title.csv** "
            "yang berisi judul berita detik.com periode Januari - Juni 2020."
        </p>
         """
    )

    df = load_raw_data()

    tab1, tab2 = st.tabs(["Sebelum Feature Engineering", "Setelah Feature Engineering"])
    with tab1:
        render_before_fe(df)
    with tab2:
        render_after_fe(df)

if __name__ == "__main__":
    run()
