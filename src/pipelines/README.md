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

`save_ciso_train_df()` uses the latest stored non-null observation as its watermark and re-fetches 72 hours of overlap by default, through the previous UTC hour. Source observations replace the value for an existing hour if EIA revised it. New hours and newly observed values for previously null rows are counted separately from revisions. A file lock serializes historical writers and a staged CSV is replaced only after the entire retrieval succeeds.

The command reports `updated` or `no_new_observations`, the source, row counts, and the latest stored UTC observation. The output location is:

```text
data/raw/historical/ciso_historical.csv
```

The seed CSV must already exist. Call `save_ciso_train_df(path=..., overlap_hours=...)` to refresh a copy or deliberately re-fetch a longer overlap. EIA hours with no reported observation remain gaps and can be filled if EIA later publishes them within the overlap.

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
python -m src.pipelines.check_inference_readiness
python -m src.pipelines.save_inference_data
```

`check_inference_readiness()` queries a 72-hour live window and returns `ready`, `no_observations`, `incomplete_context`, or `stale`. It requires the latest 25 actual observations to be exactly one hour apart before computing features. The target is the newest observation timestamp plus two hours. Once that target has arrived, the data is stale for a forward forecast. A future worker can use `result.features` as a one-row DataFrame when `result.status == "ready"` and use `result.target_timestamp` to record the forecast horizon.

`save_ciso_inference_df()` uses the same check and writes the complete validated window to `data/raw/inference/ciso_inference.csv` on `ready`. The CLI reports source, observation count, latest UTC observation and age. It returns a nonzero exit status for other readiness results or EIA errors. To select the Grid Monitor directly, set `EIA_SOURCE=dashboard` in `.env` or the worker environment. The default `auto` tries both EIA retrieval paths as described in [the ingestion module documentation](../data/README.md#ingest_datapy).

Preprocessing inference dipanggil sebagai fungsi, karena modulnya belum mempunyai blok CLI:

```bash
python -c "from src.pipelines.preprocess_inference_data import preprocess_inference_data; print(preprocess_inference_data('data/raw/inference/ciso_inference.csv'))"
```

Output adalah `data/interim/inference/processed_ciso_inference.csv` tanpa target.

### One online inference cycle

Run from the repository root while the Compose FastAPI and MLflow services are running.

```powershell
$env:INFERENCE_API_URL = "http://127.0.0.1:8000"
$env:EIA_SOURCE = "dashboard"
python -m src.pipelines.run_online_inference
```

`INFERENCE_API_URL` defaults to `http://127.0.0.1:8000`. The command sends the ten model features to `POST /predict` and prints one JSON result. It exits with code `0` on `success` and code `1` on a skipped or failed cycle. The forecast target is two hours after the observation, and an expired target yields `stale`.

For a Python caller, `run_online_inference()` returns the same result as a dataclass, with `to_dict()` for JSON serialization. `prediction_created_at` records when the client accepted the prediction, and `inference_latency_ms` measures the FastAPI request and response validation. `model_version` is the version loaded by FastAPI at startup. `/mlflow-check` exposes both the registry's current `model_version` under `champion` and FastAPI's `loaded_model_version`.

### Live verification on 2026-11-27 UTC

API v2 metadata returned HTTP 200 in 1.12 seconds and confirmed hourly UTC, `respondent`, `type`, and `value`. The CISO four-hour query with `length=1` returned HTTP 504 in 61.28 seconds. The same query through curl hit its 13-second deadline. Short Requests probes without sort or offset, for January 2025, for PJM, for monthly retail sales, and for the EIA legacy series translation all hit their roughly 12-second read limits.

EIA Grid Monitor returned 72 reported CISO demand observations for the readiness window. The latest was `2026-11-27 15:00 UTC`, 21,443 MWh, 1.40 hours old on the default-configuration check, with 25 consecutive hours and a `17:00 UTC` target. The inference save command persisted that window through the automatic fallback. An isolated historical refresh added 672 observed hours, applied five revised values, and produced zero duplicate timestamps. Seven hours had an imputed value and no reported observation in the backfill range. [EIA Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/)

## Ringkasan entry point

| Modul | Input utama | Output | CLI `python -m` |
| --- | --- | --- | --- |
| `save_train_historical_data` | CSV historis awal dan EIA | CSV raw historis diperbarui | Ya |
| `check_inference_readiness` | EIA, jendela 72 jam | Status, timestamp, dan fitur yang siap pakai | Ya |
| `save_inference_data` | EIA, jendela 72 jam | Riwayat demand siap untuk inference | Ya |
| `run_online_inference` | EIA dan FastAPI | Satu hasil prediksi live atau status kegagalan | Ya |
| `preprocess_train_data` | CSV raw historis | CSV interim dengan target t+2 | Ya |
| `preprocess_inference_data` | CSV raw inference | CSV interim tanpa target | Tidak; panggil fungsi |
| `build_train_features` | CSV interim historis tetap | CSV fitur training | Ya |

Workflow GitHub Actions menjalankan preprocessing dan build fitur setelah menarik raw dari DVC; workflow tidak memanggil ingestion EIA. Lihat [README utama](../../README.md#github-actions-ci-dan-continuous-training) dan [training model](../models/README.md).
