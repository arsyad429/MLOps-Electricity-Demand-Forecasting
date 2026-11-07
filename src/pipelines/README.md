# Data pipelines

Entry point pengumpulan data, preprocessing, dan pembentukan fitur demand CISO untuk forecasting **t+2 jam**. Jalankan semua perintah dari root repository; modul preprocessing menggunakan `Path.cwd()` sebagai dasar lokasi output.

## Prasyarat

Pasang dependency proyek dan isi `EIA_API_KEY` di environment/`.env` jika akan mengakses EIA. Pipeline historis memerlukan dataset awal `data/raw/historical/ciso_historical.csv`; pulihkan versi DVC dengan akses remote yang sesuai:

```bash
python -m pip install -r requirements.txt "dvc[s3]"
dvc pull -r dagshub data/raw/historical/ciso_historical.csv.dvc
```

Credential dan konfigurasi remote dijelaskan di [README utama](../../README.md#memulihkan-dataset-dengan-dvc). Direktori output dibuat otomatis oleh modul terkait.

## Pembaruan historis incremental

```bash
python -m src.pipelines.save_train_historical_data
```

`save_ciso_train_df()` membaca CSV yang sudah ada, mencari timestamp terakhir, lalu mengambil data mulai `last_timestamp + 1 jam` sampai jam UTC sekarang dikurangi satu jam. `fetch_incremental_data(start_timestamp, end_timestamp, max_retries=3)` membagi rentang menjadi chunk maksimal 24 jam.

Error HTTP 500/502/503/504 dan error request lain dicoba ulang, maksimal tiga percobaan per chunk dengan jeda 5 lalu 10 detik. Error HTTP lain diteruskan langsung. Setelah seluruh chunk selesai, baris baru digabung dengan data lama, dideduplikasi berdasarkan `period` dengan `keep="last"`, diurutkan, dan disimpan ke:

```text
data/raw/historical/ciso_historical.csv
```

Jika dataset sudah mutakhir atau respons baru kosong, pipeline selesai tanpa menulis ulang dataset. Tidak ada argumen rentang tahun pada `save_ciso_train_df()` dan tidak ada bootstrap otomatis bila CSV belum tersedia. Pipeline juga tidak memeriksa ulang gap atau revisi data sebelum timestamp terakhir.

## Preprocessing training

```bash
python -m src.pipelines.preprocess_train_data
```

CLI membaca file historis tetap dan menulis:

```text
data/interim/historical/processed_ciso_historical.csv
```

Untuk input lain, panggil fungsi Python:

```bash
python -c "from src.pipelines.preprocess_train_data import preprocess_train_data; print(preprocess_train_data('data/raw/historical/ciso_historical.csv'))"
```

Output memakai nama `processed_<nama-file-input>.csv`. Transformasi membersihkan waktu/nilai dan membuat `target_demand_2h` dengan mencocokkan timestamp tepat dua jam ke depan. Baris tanpa target masih ada pada output interim.

## Pembentukan fitur training

```bash
python -m src.pipelines.build_train_features
dvc add data/processed/historical/ciso_train_features.csv
```

`build_train_features()` membaca file interim dengan nama tetap, mengonversi `period` ke datetime UTC, menambahkan [fitur kalender/lag/rolling](../features/README.md), membuang seluruh baris dengan NaN, dan menulis:

```text
data/processed/historical/ciso_train_features.csv
```

Perbarui pointer DVC sesudah dataset berubah agar lineage training cocok dengan CSV aktual. Untuk menyimpan perubahan raw, jalankan juga `dvc add data/raw/historical/ciso_historical.csv`. Unggah object ke remote yang dituju dan commit pointer terkait ke Git. Pipeline ini bukan stage `dvc repro`; repository belum memiliki `dvc.yaml`.

## Data inference

```bash
python -m src.pipelines.save_inference_data
```

Pipeline meminta data dari tiga jam sebelum waktu UTC saat ini hingga jam UTC saat ini, mengurutkan hasil, lalu menyimpan satu record terbaru ke `data/raw/inference/ciso_inference.csv`. Nama file tetap dan akan ditimpa. Keterlambatan publikasi dapat menghasilkan respons kosong; kasus ini belum ditangani khusus.

Preprocessing inference dipanggil sebagai fungsi, karena modulnya belum mempunyai blok CLI:

```bash
python -c "from src.pipelines.preprocess_inference_data import preprocess_inference_data; print(preprocess_inference_data('data/raw/inference/ciso_inference.csv'))"
```

Output adalah `data/interim/inference/processed_ciso_inference.csv` tanpa target. Belum ada pipeline pembentukan fitur inference dan prediksi live; satu record terbaru belum menyediakan riwayat untuk lag 24 jam.

## Ringkasan entry point

| Modul | Input utama | Output | CLI `python -m` |
| --- | --- | --- | --- |
| `save_train_historical_data` | CSV historis awal dan EIA | CSV raw historis diperbarui | Ya |
| `save_inference_data` | EIA, jendela tiga jam | Satu record raw terbaru | Ya |
| `preprocess_train_data` | CSV raw historis | CSV interim dengan target t+2 | Ya |
| `preprocess_inference_data` | CSV raw inference | CSV interim tanpa target | Tidak; panggil fungsi |
| `build_train_features` | CSV interim historis tetap | CSV fitur training | Ya |

Workflow GitHub Actions menjalankan preprocessing dan build fitur setelah menarik raw dari DVC; workflow tidak memanggil ingestion EIA. Lihat [README utama](../../README.md#github-actions-ci-dan-continuous-training) dan [training model](../models/README.md).
