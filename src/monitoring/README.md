# Monitoring

Evaluasi produksi CISO **t+2 jam** mencocokkan prediksi yang tersimpan dengan demand EIA pada timestamp target UTC yang sama. SQLite menyimpan aktual, sumber, nomor revisi, dan waktu evaluasi per ID prediksi.

## Evaluasi prediksi live

Pipeline inferensi mencoba mengevaluasi prediksi lama pada setiap siklus. Target yang sudah lewat dan belum memiliki aktual berstatus `pending_actual`. Setelah aktual tersedia, evaluasi menghitung error dari prediksi awal. Jika EIA merevisi aktual, siklus berikutnya memperbarui hasil evaluasi dan riwayat revisi tetap tersedia di `observation_revisions`.

```powershell
python -m src.pipelines.evaluate_predictions --window-hours 168
python -m src.storage.prediction_store evaluations --limit 5
```

Perintah evaluasi memproses semua pasangan yang siap dan menampilkan ringkasan sesuai jendela target yang dipilih. Pakai `--summary-only` untuk membaca hasil yang tersimpan. Laporan memisahkan MAE dan RMSE dalam MWh, serta WAPE dalam persen, berdasarkan nama dan versi model. WAPE bernilai `null` jika total aktual nol. `evaluations_updated` menghitung evaluasi baru dan evaluasi yang diperbarui.

SQLite bermigrasi otomatis ke skema 3. Untuk volume Docker, gunakan image worker terbaru agar siklus inferensi dan perintah evaluasi memakai versi skema yang sama. Lihat [petunjuk worker Docker](../pipelines/README.md#hourly-docker-worker).

## Bukti evaluasi yang sudah tersedia

| Sumber | Informasi |
| --- | --- |
| MLflow training runs | Validation/test MAE, RMSE, WAPE (%), parameter, target/horizon, dan hash dataset |
| `config/model_metadata.json` | Snapshot registry training manual, diperbarui melalui script export |
| Metadata LK-07 yang dilacak DVC | Versi, stage, alias, parameter, metrik, dan lineage lifecycle |
| Artifact Actions `mlops-evidence-<run_id>` | Candidate, evaluation report, registration report jika tersedia, dan pointer fitur |

Evaluation gate untuk kandidat training memakai validation RMSE <= 1200 dan WAPE <= 3,0% dari `config/ci_thresholds.yaml`. Kandidat yang lolos masuk registry CI pada Staging.

## Data Drift Monitoring

Worker menyimpan payload asli untuk setiap prediksi produksi baru melalui transaksi yang sama dengan `predictions`. Exporter membaca `prediction_features` bersama timestamp observasi, timestamp inferensi, timestamp target, dan identitas model yang tersimpan di `predictions`. Jendela dibatasi oleh timestamp **observasi** UTC. Satu jam observasi dihitung paling banyak sekali per versi model. Prediksi historis tanpa payload, permintaan langsung ke FastAPI untuk load test, serta model lain berada di luar sampel. Data aktual target belum diperlukan.

Exporter membaca nama dan **versi yang sudah dimuat FastAPI** dari `/mlflow-check`, lalu mencocokkannya dengan artefak lokal di `references/`. File referensi champion saat ini adalah [CISO-Demand-Forecaster-Compose v1](references/CISO-Demand-Forecaster-Compose__1.json). Model itu diimpor dari `CISO-Demand-Forecaster-CI` v2. Baseline memakai **47.352 baris training** dari objek DVC `0751dbec244c880063464f795a848774`, sesuai pointer arsip MLflow run `df79fb70265d43238bd058129bc41849`. Build memeriksa checksum, signature model, dan split kronologis dengan purge dua jam. Baris validation dan test dikeluarkan. Docker menyalin artefak JSON bersama `src`, sehingga proses scrape tidak membutuhkan DVC, dataset CSV, atau kredensial registry sumber.

Untuk membuat referensi saat model berganti, jalankan dari root dengan venv proyek, akses MLflow Compose, kredensial DagsHub di `.env.compose`, dan remote DVC yang sudah dikonfigurasi. Pakai versi yang dilaporkan sebagai `loaded_model_version`. Jika objek yang persis sama sudah tersedia lokal, tambahkan `--csv <path>`. Perintah hanya menulis artefak untuk versi yang dipilih dan menolak penggantian artefak dengan isi berbeda pada path yang sama.

```powershell
python -m scripts.build_drift_reference --model-name CISO-Demand-Forecaster-Compose --model-version 1
```

### Skor dan konfigurasi

PSI menjumlahkan `(aktual - referensi) × ln(aktual / referensi)` atas proporsi setiap bin. Default memakai hingga 10 bin referensi. Fitur kontinu memakai batas kuantil dan batas berulang digabung. Fitur diskret dengan kategori sedikit memakai pemisah antar nilai. Bin terluar mencakup nilai di luar rentang training, dan fitur konstan memakai bin di bawah, sama dengan, serta di atas nilai tersebut. Smoothing simetris sebesar `0.0001` pada proporsi menghindari log nol. Nilai hilang atau non-finite tidak masuk hitungan fitur terkait. Payload dengan kolom berbeda ditolak dan tercatat pada metrik.

Jumlah bin diatur dengan `--bins` saat membangun referensi. Untuk eksperimen dengan bin berbeda, gunakan `--output-dir` terpisah. Exporter selalu memakai batas yang disimpan pada artefak, sehingga bin tetap konsisten antar-scrape.

| Env | Default |
| --- | --- |
| `DRIFT_WINDOW_HOURS` | `168` |
| `DRIFT_MIN_PRODUCTION_SAMPLES` | `100` nilai finite dari jam observasi unik, per fitur |
| `DRIFT_MODERATE_THRESHOLD` | `0.10` |
| `DRIFT_SIGNIFICANT_THRESHOLD` | `0.25` |
| `DRIFT_REFRESH_SECONDS` | `60` |
| `DRIFT_MODEL_CHECK_URL` | `http://api-service:8000/mlflow-check` di Compose, atau `http://127.0.0.1:8000/mlflow-check` di luar Docker |

Status `-1` berarti **INSUFFICIENT_DATA**, termasuk saat model aktif, artefak, atau data yang diperlukan belum tersedia. PSI di bawah `0.10` mendapat `0` (**STABLE**), rentang `[0.10, 0.25)` mendapat `1` (**MODERATE_DRIFT**), dan PSI `≥ 0.25` mendapat `2` (**SIGNIFICANT_DRIFT**) dengan konfigurasi default. Status keseluruhan mengikuti tingkat drift tertinggi. Jika sebagian fitur tidak memiliki skor dan semua fitur terukur stabil, status keseluruhan tetap `INSUFFICIENT_DATA`. Setelah tiap penyegaran, skor yang sudah tidak berlaku dikeluarkan dari scrape. Ambang dan jumlah minimum adalah heuristik awal. [Studi sifat statistik PSI](https://files.wmich.edu/s3fs-public/attachments/u730/2022/PSIfinal.pdf)

PSI dijalankan untuk enam fitur demand dan empat fitur kalender yang numerik. Baseline mencakup beberapa tahun, sedangkan jendela tujuh hari biasanya menempati satu atau dua bulan. Fitur `month` dapat mendominasi status keseluruhan hanya karena perbedaan cakupan kalender. Perubahan cuaca, hari libur, komposisi jam, dan jadwal worker juga bisa menaikkan PSI. Snapshot berjam-jam berdekatan saling berkorelasi, terutama lag dan rolling mean. Interpretasikan skor sebagai sinyal perubahan distribusi. Analisis model decay tetap memakai MAE, RMSE, dan WAPE dari evaluasi saat aktual target tersedia. Langkah kalibrasi berikutnya adalah membandingkan jendela historis yang sepadan secara musiman.

### Prometheus dan Grafana

Metrik berikut berupa Gauge dari job `ciso-production`. Semua seri membawa `model_name` dan `model_version`. Seri `ciso_feature_*` juga membawa `feature`. Gunakan `{job="ciso-production"}` setelah nama metrik untuk memeriksa satu ekspresi pada Prometheus.

| Metrik | Arti |
| --- | --- |
| `ciso_drift_status` | Status keseluruhan |
| `ciso_feature_drift_status` | Status per fitur dan kecukupan datanya |
| `ciso_feature_psi` | Skor untuk setiap fitur yang memenuhi syarat |
| `ciso_drifted_features`, `ciso_significant_drift_features` | Jumlah fitur terukur di atas ambang masing-masing |
| `ciso_drift_reference_samples`, `ciso_drift_production_samples` | Jumlah training dan payload unik yang cocok dengan schema |
| `ciso_feature_drift_production_samples`, `ciso_drift_min_production_samples` | Jumlah nilai finite per fitur dan batas minimum |
| `ciso_drift_evaluated_features`, `ciso_drift_invalid_payloads` | Cakupan skor dan jumlah payload dengan schema berbeda |
| `ciso_drift_max_psi` | Maksimum dari skor saat ini |
| `ciso_drift_evaluation_timestamp_seconds` | Waktu pengecekan terakhir, termasuk hasil insufficient |
| `ciso_drift_baseline_available`, `ciso_drift_sqlite_read_success` | Ketersediaan baseline aktif dan hasil pembacaan SQLite |
| `ciso_drift_moderate_threshold`, `ciso_drift_significant_threshold` | Ambang konfigurasi untuk garis tren |

Di [dashboard LK-11](http://127.0.0.1:3000/d/792ce94f-4dfa-4c12-8fb9-ff28d0063d14/mlops-electricity-demand-forecasting-dashboard), buka **Data Drift Monitoring**. Periksa jumlah sampel dan status per fitur sebelum membaca skor. Skor dan penghitung drift kosong jika belum ada fitur yang dapat dinilai. Grafik memperlihatkan riwayat PSI yang benar-benar diterbitkan serta batas konfigurasi saat skor tersedia.

```powershell
docker compose --profile online config -q
docker compose --profile online up -d --build --no-deps production-metrics
$query = [uri]::EscapeDataString('ciso_drift_status{job="ciso-production"}')
(Invoke-RestMethod "http://127.0.0.1:9090/api/v1/query?query=$query").data.result
```

Grafana membaca ulang file provisioning dashboard setiap 30 detik. Prometheus tetap membaca job dan target yang sama. Jika estimator baru memiliki referensi yang baru dibuat, rebuild exporter agar JSON masuk ke image.

### Pengujian dan demonstrasi

```powershell
python -m pytest -q
python -m scripts.demo_data_drift
```

Demo menjalankan detector dengan payload sintetis pada tiga SQLite sementara dan model check loopback. Output yang diharapkan adalah `STABLE` dengan 100 sampel, `SIGNIFICANT_DRIFT` dengan 100 sampel, dan `INSUFFICIENT_DATA` dengan 40 sampel. Demo membersihkan database sementara ketika selesai.

## Retraining

Retraining yang tersedia sekarang dipicu perubahan kode/push ke `main` atau `workflow_dispatch` pada ref `main`. Retraining terjadwal, trigger drift, dan promosi Production otomatis belum tersedia. Detail perilaku saat ini tersedia pada [README models](../models/README.md), [README scripts](../../scripts/README.md), dan [README utama](../../README.md#github-actions-ci-dan-continuous-training).
