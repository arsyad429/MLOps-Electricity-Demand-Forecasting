# Model training dan registry

Folder ini berisi training **CISO demand t+2 jam** serta CLI lifecycle MLflow LK-07. Jalankan seluruh perintah dari root repository setelah memasang dependency, menyiapkan dataset, dan menjalankan server MLflow sesuai [README utama](../../README.md).

## Dataset dan fitur

`train.py` membaca file tetap berikut:

- CSV: `data/processed/historical/ciso_train_features.csv`.
- Pointer lineage: `data/processed/historical/ciso_train_features.csv.dvc`.
- Timestamp: `period`, dikonversi ke datetime UTC.
- Target: `target_demand_2h`, demand pada timestamp dua jam setelah observasi.

Urutan `FEATURE_COLUMNS` adalah:

```text
value, lag_1h, lag_2h, lag_24h, rolling_mean_3h,
rolling_mean_24h, hour, day_of_week, month, is_weekend
```

Loader memeriksa file dan kolom wajib, menolak timestamp duplikat, mengonversi nilai numerik, membuang NaN/infinity pada kolom wajib, lalu mengurutkan waktu. Baris juga disaring dengan `period.diff(24) == 24 jam` sebagai pemeriksaan konteks hourly; ini bukan imputasi atau perbaikan gap. Minimal 100 baris harus tersisa.

Hash dataset dibaca dari pointer DVC, bukan dihitung ulang dari CSV oleh loader. Setelah membangun ulang fitur, jalankan `dvc add data/processed/historical/ciso_train_features.csv` sebelum training. Langkah persiapan tersedia di [README pipelines](../pipelines/README.md).

## Training manual

```bash
python -m src.models.train --trials 10 --tracking-uri http://127.0.0.1:5000
```

| Opsi | Default | Makna |
| --- | --- | --- |
| `--trials` | 50 | Trial Optuna per algoritma; minimal 1 |
| `--tracking-uri` | `MLFLOW_TRACKING_URI` atau `http://127.0.0.1:5000` | Tracking server |
| `--ci` | Tidak aktif | Menunda registrasi dan menulis kandidat untuk evaluation gate |

Alur training:

1. Split kronologis sekitar 70% train, 15% validation, 15% test; baris train/validation dengan label t+2 yang mencapai batas split berikutnya dibuang.
2. Tuning XGBoost dan LightGBM melalui Optuna TPE dengan seed 42, meminimalkan validation RMSE.
3. Fit kandidat terbaik masing-masing algoritma pada train dan log hasil validasi.
4. Pilih pemenang berdasarkan validation RMSE terendah.
5. Evaluasi kedua kandidat pada test hanya setelah pemilihan, untuk pelaporan.

MAE dan RMSE dilaporkan dalam skala demand. WAPE dihitung sebagai `sum(abs(actual - predicted)) / sum(abs(actual)) * 100`; jika denominator nol, hasilnya NaN. Hyperparameter, metrics, tags target/horizon/hash dataset, model artifacts, input example, signature, pointer DVC, dan daftar fitur dicatat ke eksperimen `CISO_Demand_Forecasting_2h_Optuna`.

Search space aktual ditentukan oleh `suggest_params`: `n_estimators` 150-600 (step 50), learning rate 0,01-0,2, dan `max_depth` 3-10, beserta parameter sampling dan parameter khusus tiap algoritma. `config/model.yaml` belum mengatur tuning atau daftar fitur runtime.

Mode manual mendaftarkan kedua kandidat ke **`CISO-Demand-Forecaster`**, lalu mengarahkan alias **`champion`** ke versi pemenang. Mode ini tidak menjalankan evaluation gate CI dan tidak mengatur legacy stage Production.

## Training dalam mode CI

```bash
python -m src.models.train --trials 3 --ci --tracking-uri http://127.0.0.1:5000
python scripts/ci_gate.py evaluate
```

Kandidat tetap dipilih berdasarkan validation RMSE. `--ci` mencatat model ke MLflow tanpa mendaftarkan versi registry atau mengubah alias, lalu menulis `ci_output/candidate.json` berisi algoritma, run ID, model URI, hash DVC, Git SHA (`GITHUB_SHA`, atau `local`), dan metrik validation/test.

Gate menggunakan `config/ci_thresholds.yaml`: validation RMSE <= 1200 dan WAPE <= 3,0%, keduanya finite. Setelah PASS, workflow mengunggah dataset fitur ke DagsHub dan menjalankan:

```bash
python scripts/ci_gate.py register
```

Registrasi memerlukan `MLFLOW_TRACKING_URI` pada environment; `--tracking-uri` dari command training tidak otomatis mengatur environment command berikutnya. Script memeriksa kecocokan run ID dan Git SHA, kemudian mendaftarkan kandidat ke **`CISO-Demand-Forecaster-CI`**, stage **Staging**, alias **`staging`**. Tidak ada promosi Production otomatis. Lihat [README scripts](../../scripts/README.md) dan [alur Actions](../../README.md#github-actions-ci-dan-continuous-training).

## Lifecycle LK-07 (`registry.py`)

Registry lifecycle **`CISO-Demand-Forecaster-LK07`** terpisah dari training manual dan CI. Default tracking URI CLI ini adalah `http://127.0.0.1:5000`; gunakan `--tracking-uri` sebelum subcommand untuk endpoint lain.

Mulai dengan memeriksa registry yang sudah ada:

```bash
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 status
```

Untuk inisialisasi pada registry LK-07 yang masih kosong, server harus sudah memiliki model sumber **`CISO-Demand-Forecaster/3`**, run asal, artifacts, dan dataset terkait:

```bash
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 register-v1
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 register-v2
```

`register-v1` menyalin artifact versi sumber tanpa retraining dan menolak registry LK-07 yang sudah memiliki versi. Jika tag hash DVC model sumber tersedia, nilainya harus cocok dengan pointer workspace. `register-v2` memerlukan v1, menolak bila v2 sudah ada, lalu melakukan retraining XGBoost dengan mengubah hanya `n_estimators`: dikurangi 50 jika nilai awal >100, selainnya ditambah 50. Nilai parameter lain diambil dari run sumber.

Promosi dilakukan berurutan dan secara eksplisit setelah menilai hasil model:

```bash
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 promote --version 2 --stage Staging
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 promote --version 2 --stage Production
```

Promosi Staging mengatur alias `staging`; promosi Production mengatur alias `champion` dan mencoba menghapus alias `staging` bila menunjuk versi yang sama. CLI tetap menggunakan API legacy stage untuk kebutuhan lifecycle ini, dengan `archive_existing_versions=False`. Promosi tidak memiliki evaluation gate otomatis.

### Metadata dan verifikasi

```bash
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 export
dvc add artifacts/lk07/model_lineage.json
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 verify
```

`export` menulis versi, stage, alias, run, parameter, metrik, feature columns, dan hash dataset ke `artifacts/lk07/model_lineage.json`. Git melacak pointer `.dvc`; isi JSON memerlukan cache/remote DVC. Unggah artifact ke remote yang dipilih dan commit pointer untuk menyimpan versi baru.

`verify` memerlukan alias `champion` dengan legacy stage Production. Model dimuat melalui `models:/CISO-Demand-Forecaster-LK07@champion` dan URI stage Production, lalu prediksi keduanya dibandingkan pada baris terakhir dataset fitur historis. Pemeriksaan ini adalah smoke test historis; belum ada layanan prediksi live.

## Membaca snapshot model

| Sumber | Isi snapshot yang diperiksa |
| --- | --- |
| `config/model_metadata.json` | Registry manual, XGBoost v3, alias `champion` |
| Metadata lokal LK-07 | v1 dari model manual v3; v2 dengan `n_estimators` 600 -> 550, Production dan alias `champion` |
| `ci_output/` | Evidence per eksekusi CI, bukan snapshot Production LK-07 |

Pada snapshot LK-07, MAE/WAPE validasi v2 lebih rendah (715,19 dan 2,714%) daripada v1 (720,93 dan 2,736%), tetapi RMSE v2 sedikit lebih tinggi (1.030,16 dibanding 1.028,97). Promosi lifecycle tersebut tidak berarti v2 memenangkan aturan validation RMSE yang digunakan training otomatis.

Snapshot tidak memperbarui dirinya sendiri dan tidak menjamin kondisi registry live. Server MLflow baru juga tidak otomatis berisi versi yang disebutkan. [Script export metadata](../../scripts/README.md) mengekspor registry manual, sedangkan subcommand `registry export` mengekspor lifecycle LK-07.
