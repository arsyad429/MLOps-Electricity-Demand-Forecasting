# Notebooks

Folder ini berisi [`api_check.ipynb`](api_check.ipynb), notebook eksplorasi akses EIA API. Notebook memuat environment dengan `python-dotenv` dan menggunakan `requests` untuk memeriksa respons API, termasuk route `electricity/rto/region-data`.

## Menjalankan

Dari root repository, setelah memasang dependency dan mengisi `EIA_API_KEY` di environment/`.env`:

```bash
jupyter notebook notebooks/api_check.ipynb
```

Pilih kernel dari virtual environment proyek. Cell yang memanggil API memerlukan koneksi internet dan credential valid. Periksa source dan output sebelum membagikan notebook agar API key atau URL request berisi credential tidak ikut tersimpan.

## Hubungan dengan pipeline

Notebook merupakan eksplorasi manual, bukan entry point automated training atau sumber status registry terbaru. Alur yang digunakan saat ini tersedia sebagai modul Python:

- [Data dan preprocessing](../src/data/README.md): client EIA dan target `target_demand_2h`.
- [Pipelines](../src/pipelines/README.md): ingestion incremental, preprocessing, dan penyimpanan fitur.
- [Features](../src/features/README.md): fitur kalender California, lag, dan rolling.
- [Models](../src/models/README.md): tuning Optuna, evaluasi, tracking, dan lifecycle registry.

Gunakan modul tersebut untuk menjalankan ulang pipeline; tidak perlu menyalin cell notebook ke workflow CI. Folder ini dapat menampung analisis EDA atau eksperimen tambahan, tetapi belum ada notebook EDA/model terpisah pada file yang dilacak saat ini.
