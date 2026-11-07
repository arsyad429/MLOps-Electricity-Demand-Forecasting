# Feature engineering

[`feature_engineering.py`](feature_engineering.py) menyediakan `add_features(df)`, yang menyalin DataFrame lalu menambahkan fitur kalender, lag, dan rolling mean. Fungsi ini sudah digunakan oleh pipeline training dan automated tests.

## Input dan output

Input harus memiliki `period` bertipe datetime dengan timezone serta `value` numerik, terurut dari waktu paling awal. Timestamp disimpan dalam UTC, sementara fitur kalender dihitung setelah konversi ke `America/Los_Angeles`, termasuk perubahan daylight saving time.

| Kolom | Perhitungan |
| --- | --- |
| `hour` | Jam lokal California, 0-23 |
| `day_of_week` | Hari lokal, Senin=0 sampai Minggu=6 |
| `month` | Bulan lokal, 1-12 |
| `is_weekend` | 1 jika `day_of_week >= 5`, selainnya 0 |
| `lag_1h` | `value.shift(1)` |
| `lag_2h` | `value.shift(2)` |
| `lag_24h` | `value.shift(24)` |
| `rolling_mean_3h` | Rata-rata tiga baris, termasuk baris saat ini |
| `rolling_mean_24h` | Rata-rata 24 baris, termasuk baris saat ini |

Model memakai sembilan fitur tambahan tersebut bersama `value`, sehingga totalnya sepuluh fitur. Daftar dan urutan input model ditetapkan oleh `FEATURE_COLUMNS` di [`../models/train.py`](../models/train.py).

## Pemakaian

```python
import pandas as pd
from src.features.feature_engineering import add_features

df = pd.read_csv("data/interim/historical/processed_ciso_historical.csv")
df["period"] = pd.to_datetime(df["period"], utc=True)
features = add_features(df)
```

Untuk menyimpan dataset training siap pakai, jalankan dari root repository:

```bash
python -m src.pipelines.build_train_features
```

Pipeline membaca file interim tersebut, memanggil `add_features`, membuang baris dengan NaN, dan menulis `data/processed/historical/ciso_train_features.csv`. Target `target_demand_2h` dibuat sebelumnya oleh preprocessing, bukan oleh fungsi fitur.

## Batas penggunaan

Lag dan rolling menggunakan posisi baris, sehingga penamaan jam mengasumsikan interval hourly yang lengkap. `add_features` tidak mengurutkan, memvalidasi gap, atau membuang NaN sendiri. Loader training menambahkan pemeriksaan konteks 24 jam, tetapi tidak memperbaiki gap.

Lag 24 baris memerlukan minimal 25 observasi berurutan agar baris terbaru memiliki seluruh fitur. Satu record dari pipeline ingestion inference belum cukup untuk prediksi live. Daftar lag dan rolling di `config/model.yaml` belum dibaca oleh fungsi ini; perilaku aktual mengikuti kode di atas.

Lihat [pipeline data](../pipelines/README.md), [training](../models/README.md), dan [pengujian](../../tests/README.md).
