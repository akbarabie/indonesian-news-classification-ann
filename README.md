# Folder saved_model

Folder ini sengaja dikosongkan di deliverable ini. Sebelum menjalankan aplikasi
(baik lokal maupun lewat Docker), salin 6 file artefak hasil Model Saving dari
notebook utama (`P2G7_Muhammad_Akbar_Suharbi.ipynb`, section 12) ke folder ini:

- model_2_baseline.keras
- model_2b_numfilters128.keras
- model_2c_dropout03.keras
- model_3_word2vec.keras
- tokenizer.pkl
- label_encoder.pkl

Tanpa keenam file ini, menu Prediksi Kategori Berita tidak akan bisa jalan,
dan bagian 5.7.3 - 5.7.4 di menu EDA (OOV rate dan wordcloud long tail) juga
butuh tokenizer.pkl.
