# Data Pipelines

Folder ini berisi entry point pengumpulan data dan fungsi preprocessing untuk data training serta inference. Semua command harus dijalankan dari root repository agar import `src` dan lokasi output konsisten.

## Prasyarat

1. Install dependency dengan `python -m pip install -r requirements.txt`.
2. Salin `.env.example` menjadi `.env` dan isi `EIA_API_KEY`.
3. Buat folder output berikut:

```powershell
New-Item -ItemType Directory -Force data/raw/historical, data/raw/inference
```

Pada Bash, gunakan `mkdir -p data/raw/historical data/raw/inference`.

## Pengumpulan Data Historis

```powershell
python -m src.pipelines.save_train_historical_data
```

`save_train_historical_data.py` mengambil data CISO per tahun, mulai 2019 sampai tahun UTC saat ini secara default, menggabungkan seluruh respons, lalu menyimpan satu file:

```text
data/raw/historical/ciso_historical_<timestamp-UTC>.csv
```

Untuk membatasi rentang tahun (inklusif):

```powershell
python -c "from src.pipelines.save_train_historical_data import save_ciso_train_df; save_ciso_train_df(2022, 2024)"
```

## Pengumpulan Data Inference

```powershell
python -m src.pipelines.save_inference_data
```

`save_inference_data.py` meminta data mulai jam UTC saat ini dan menyimpan hasil ke:

```text
data/raw/inference/ciso_inference_<timestamp-UTC>.csv
```

Data EIA dapat memiliki jeda publikasi, sehingga respons untuk jam terbaru mungkin belum berisi record.

## Preprocessing

File preprocessing menyediakan fungsi Python dan belum mempunyai CLI. Gunakan file raw yang dihasilkan pipeline sebagai input.

Training:

```powershell
python -c "from src.pipelines.preprocess_train_data import preprocess_train_data; print(preprocess_train_data(r'data/raw/historical/NAMA_FILE.csv'))"
```

Hasil disimpan di `data/interim/historical/processed_NAMA_FILE.csv` dan memiliki kolom `target_demand_1h`.

Inference:

```powershell
python -c "from src.pipelines.preprocess_inference_data import preprocess_inference_data; print(preprocess_inference_data(r'data/raw/inference/NAMA_FILE.csv'))"
```

Hasil disimpan di `data/interim/inference/processed_NAMA_FILE.csv` tanpa kolom target.

## Ringkasan File

| File | Kegunaan | Dapat dijalankan dengan `python -m` |
| --- | --- | --- |
| `save_train_historical_data.py` | Unduh dan simpan data historis | Ya |
| `save_inference_data.py` | Unduh dan simpan data terbaru | Ya |
| `preprocess_train_data.py` | Bersihkan data dan buat target t+1 | Belum; panggil fungsinya |
| `preprocess_inference_data.py` | Bersihkan data inference | Belum; panggil fungsinya |

Detail fungsi koneksi EIA dan transformasi data tersedia di [`../data/README.md`](../data/README.md).
