# Frontend

Folder ini disiapkan untuk dashboard **React + Vite** yang menampilkan demand CISO dan forecast **t+2 jam**. Saat ini belum ada `package.json`, source aplikasi, komponen UI, atau integrasi API di folder ini.

## Kondisi integrasi

Pipeline data, feature engineering, training, dan registrasi model sudah tersedia melalui Python dan GitHub Actions. Backend REST untuk konsumsi dashboard masih direncanakan; lihat [README backend](../backend/README.md). Verifikasi prediksi yang ada menggunakan sampel historis melalui CLI model registry.

Dashboard mendatang dapat menampilkan demand aktual dan prediksi, waktu observasi/horizon, model yang digunakan, metrik evaluasi, serta status pipeline. Metadata perlu dibedakan antara snapshot training manual, lifecycle LK-07, dan kandidat CI di Staging. File `config/model_metadata.json` tidak otomatis mencerminkan hasil CI terbaru.

## Environment pengembangan

Devcontainer menyediakan Node.js 20 dan meneruskan port 5173, tetapi belum menjalankan service frontend. Karena proyek frontend belum diinisialisasi, belum tersedia command `npm install`, `npm run dev`, atau build untuk folder ini.

Untuk alur yang sudah dapat dijalankan, gunakan [README utama](../README.md), [pipeline data](../src/pipelines/README.md), dan [training model](../src/models/README.md).
