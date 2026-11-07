# Backend

Folder ini disiapkan untuk layanan **FastAPI** yang akan menyajikan prediksi demand CISO **t+2 jam**. Saat ini hanya dokumentasi yang tersedia: belum ada aplikasi ASGI, endpoint REST, dependency injection, atau konfigurasi deployment backend.

## Komponen yang sudah tersedia

Training dan model lifecycle sudah diimplementasikan di [`src/models`](../src/models/README.md). Tiga registry mempunyai peran berbeda: `CISO-Demand-Forecaster` untuk training manual, `CISO-Demand-Forecaster-LK07` untuk lifecycle eksperimen, dan `CISO-Demand-Forecaster-CI` untuk kandidat yang lolos gate dan masuk Staging.

Pemeriksaan model yang tersedia saat ini dilakukan melalui CLI:

```bash
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 status
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 verify
```

Jalankan dari root repository. Command memerlukan MLflow server dengan registry LK-07; `verify` juga memerlukan champion berstatus Production dan dataset fitur historis lokal. Hasilnya adalah smoke test historis, bukan endpoint prediksi live.

## Rencana integrasi

Layanan serving perlu memuat versi/alias model yang ditentukan, memvalidasi sepuluh input sesuai `FEATURE_COLUMNS`, membangun fitur inference dari riwayat demand, dan mengembalikan timestamp serta horizon prediksi. Pengambilan satu record EIA terbaru belum cukup untuk lag 24 jam. Kandidat Staging dari CI juga belum dipromosikan otomatis ke Production.

`fastapi` dan `uvicorn` sudah tercantum dalam `requirements.txt`. Port 8000 diteruskan oleh devcontainer, tetapi belum ada module aplikasi untuk dijalankan dengan Uvicorn. `DATABASE_URL` pada `.env.example` masih placeholder. Setup proyek dan batas implementasi tercantum pada [README utama](../README.md).
