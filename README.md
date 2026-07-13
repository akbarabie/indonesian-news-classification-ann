# Deployment - Klasifikasi Kategori Berita Indonesia

Folder ini berisi source code webapps Streamlit untuk project Graded Challenge 7.

## Struktur File

```
deployment/
├── app.py                   entry point, navigasi antar menu
├── eda.py                   menu Exploratory Data Analysis
├── prediction.py             menu Prediksi Kategori Berita
├── requirements.txt
├── detik_news_title.csv      dataset, dipakai oleh menu EDA
└── saved_model/               taruh 6 file artefak model di sini (lihat PLACEHOLDER.md)
```

## Sebelum Menjalankan

Pastikan folder `saved_model/` sudah berisi keenam file artefak hasil training
dari notebook utama0. Tanpa file ini, menu Prediksi dan sebagian menu EDA (5.7.3 - 5.7.4) tidak akan
bisa jalan.
