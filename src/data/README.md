# Data modules

Modul reusable untuk mengambil demand listrik CISO dari EIA dan menyiapkan data hourly untuk prediksi **t+2 jam**. Jalankan contoh dari root repository setelah memasang [dependency proyek](../../README.md#persiapan-environment).

## `ingest_data.py`

`get_ciso_EIA_data(start_date=None, end_date=None, train=True)` membaca `EIA_API_KEY` melalui environment/`.env` dan mengembalikan JSON respons API. Request menggunakan route `electricity/rto/region-data/data/`, frekuensi hourly, respondent `CISO`, type `D`, field `value`, urutan `period` ascending, dan batas 5.000 record tanpa pagination. Timeout setiap request adalah 120 detik; error HTTP diteruskan sebagai exception.

Dalam mode training, kedua batas waktu wajib diberikan:

```python
import pandas as pd
from src.data.ingest_data import get_ciso_EIA_data

result = get_ciso_EIA_data(
    start_date="2024-01-01T00",
    end_date="2024-01-07T23",
)
df = pd.DataFrame(result["response"]["data"])
```

Pilih rentang yang sesuai dengan batas record; fungsi ini tidak otomatis memecah request panjang. Pemecahan chunk dan retry tersedia pada pipeline historis.

Dalam mode `train=False`, batas waktu dihitung ulang menjadi tiga jam sebelum waktu UTC sekarang hingga jam UTC sekarang. Pipeline inference kemudian memilih satu record terbaru. Respons kosong belum memiliki penanganan khusus pada pipeline tersebut.

`save_raw_data(df, prefix)` membuat direktori induk dan menyimpan CSV ke `data/raw/<prefix>.csv`, tanpa index. Nama file tetap dan akan ditimpa; contoh prefix `historical/ciso_historical` atau `inference/ciso_inference`.

## `preprocess.py`

Nama fungsi publik mengikuti implementasi: `formating_ciso_data(df, train=True)`.

1. Mengubah `period` menjadi datetime dengan timezone UTC.
2. Mengubah `value` menjadi numerik; nilai yang tidak bisa dikonversi menjadi NaN.
3. Membuang baris duplikat penuh dan baris dengan nilai kosong, kemudian mengurutkan `period` dan mereset index.
4. Pada mode training, memanggil `add_target_2h(df)` untuk menambahkan `target_demand_2h`.

`add_target_2h` menggabungkan demand berdasarkan timestamp tepat `period + 2 jam`. Target tidak dibentuk dengan menggeser dua baris, sehingga gap waktu tidak dianggap sebagai dua jam. Baris tanpa pasangan timestamp tetap memiliki target NaN dan dibuang saat membangun fitur training.

```python
from src.data.preprocess import formating_ciso_data

train_df = formating_ciso_data(df.copy(), train=True)
inference_df = formating_ciso_data(df.copy(), train=False)
```

Mode inference tidak menambahkan target. Fungsi memodifikasi kolom input sebelum beberapa transformasi berikutnya; gunakan `.copy()` jika DataFrame asli ingin dipertahankan. Preprocessing belum melakukan resampling, imputasi, atau deduplikasi khusus berdasarkan timestamp yang nilainya berbeda.

## Entry point

Modul data bukan CLI. Gunakan [pipeline](../pipelines/README.md) berikut:

```bash
python -m src.pipelines.save_train_historical_data
python -m src.pipelines.preprocess_train_data
python -m src.pipelines.build_train_features
```

Pipeline historis memerlukan CSV awal yang sudah ada; pulihkan melalui DVC terlebih dahulu. Detail versioning dan remote tersedia di [README utama](../../README.md#memulihkan-dataset-dengan-dvc).
