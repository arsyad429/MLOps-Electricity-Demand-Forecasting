# Data modules

Modul reusable untuk mengambil demand listrik CISO dari EIA dan menyiapkan data hourly untuk prediksi **t+2 jam**. Jalankan contoh dari root repository setelah memasang [dependency proyek](../../README.md#persiapan-environment).

## `ingest_data.py`

`fetch_ciso_observations(start, end, config=None)` takes inclusive UTC hour bounds and returns a DataFrame sorted by `period`, with UTC timestamps and numeric MWh. It validates the response schema, series, units, timestamp uniqueness, and numeric values. A published null is an unavailable hour. An empty DataFrame is a successful query with no actual observations.

```python
from src.data.ingest_data import fetch_ciso_observations

observations = fetch_ciso_observations(
    "2026-11-24T16",
    "2026-11-27T16",
)
print(observations.attrs["source"], observations[["period", "value"]].tail(1))
```

`EIA_SOURCE=auto` starts with API v2 and tries EIA’s Grid Monitor after a temporary failure. `api` and `dashboard` select a single source. API v2 uses `EIA_API_KEY`, filters `CISO` and `D`, and pages through stable timestamps using `offset`, `length`, and `response.total`. The Grid Monitor uses bounded date windows and `REPORTED_VAL`, which excludes hours where EIA has only an imputed value. The `source` attribute identifies the endpoint that supplied the result. [EIA API documentation](https://www.eia.gov/opendata/documentation.php) · [EIA-930 data and reporting conventions](https://www.eia.gov/electricity/930-content/about.html)

Configure each attempt with `EIA_CONNECT_TIMEOUT` (5 seconds), `EIA_READ_TIMEOUT` (12 seconds), and `EIA_MAX_ATTEMPTS` (3), or pass an `EIAConfig`. Retries apply to transport failures and HTTP 429, 500, 502, 503, and 504. Backoff is exponential with jitter and honors `Retry-After` within the retry budget. The client raises `EIATemporaryError`, `EIARequestError`, or `EIAResponseError` for temporary, request/configuration, and invalid-response failures respectively. Messages include safe context such as UTC bounds and HTTP status.

`get_ciso_EIA_data(...)` retains the `response.data` access pattern for existing callers. Its `train=False` shortcut covers three hours. Use the inference pipeline below for the full model feature window. `save_raw_data(df, prefix)` writes `data/raw/<prefix>.csv` by replacing a staged file on the same filesystem.

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
