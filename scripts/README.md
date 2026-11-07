# Scripts operasional

Folder ini berisi evaluation gate CI dan export metadata registry manual. Jalankan command dari root repository dengan dependency Python terpasang. Pipeline pengumpulan/preprocessing tetap berada di [`src/pipelines`](../src/pipelines/README.md).

## `ci_gate.py evaluate`

```bash
python scripts/ci_gate.py evaluate
```

Input:

- `ci_output/candidate.json`, dihasilkan oleh `python -m src.models.train --ci`.
- `config/ci_thresholds.yaml`, konfigurasi threshold validation.

Untuk konfigurasi saat ini, kandidat harus memiliki validation RMSE <= **1200** dan validation WAPE <= **3,0%**. Threshold harus positif dan finite; nilai metrik kandidat juga harus finite. Semua pemeriksaan yang dikonfigurasi harus lolos, dan setidaknya satu threshold harus tersedia.

Output `ci_output/evaluation.json` berisi status PASS/FAIL, run ID, Git SHA, nilai aktual, threshold, dan hasil setiap pemeriksaan. Jika metrik gagal, script menulis laporan lalu keluar dengan status gagal. Metrik test tidak dipakai oleh gate.

## `ci_gate.py register`

```bash
python scripts/ci_gate.py register
```

Script memerlukan candidate, evaluation berstatus PASS, dan environment `MLFLOW_TRACKING_URI`. Run ID dan Git SHA laporan evaluasi harus cocok dengan kandidat. Jika `GITHUB_SHA` tersedia, kandidat juga harus berasal dari commit tersebut.

Setelah pemeriksaan, script mendaftarkan model ke `CISO-Demand-Forecaster-CI`, mencatat tags algoritma/hash dataset/Git SHA/status validasi, memindahkannya ke legacy stage **Staging**, dan mengatur alias **`staging`**. Hasil dicatat pada `ci_output/registration.json`. Pengulangan command yang berhasil dapat membuat versi registry baru.

Pada tracking server yang memerlukan autentikasi, siapkan `MLFLOW_TRACKING_USERNAME` dan `MLFLOW_TRACKING_PASSWORD` di environment. Workflow mengisinya dari Actions variables/secrets.

## Reproduksi langkah CI

Setelah memulihkan dataset, membangun fitur, memperbarui pointer DVC, dan menyiapkan environment MLflow:

```bash
python -m src.models.train --trials 3 --ci
python scripts/ci_gate.py evaluate
dvc push -r dagshub data/processed/historical/ciso_train_features.csv.dvc
python scripts/ci_gate.py register
```

Jalankan langkah berikutnya hanya bila langkah sebelumnya berhasil. Pada CI, `MLFLOW_TRACKING_URI` tersedia di environment seluruh job. Workflow menyimpan `ci_output/` dan pointer dataset sebagai artifact `mlops-evidence-<run_id>`; file tersebut tidak otomatis di-commit ke Git. Lihat [workflow utama](../.github/workflows/mlops-automation.yaml) dan [README models](../src/models/README.md).

## `export_model_metadata.py`

```bash
python scripts/export_model_metadata.py
```

Script ini menghubungi **`http://127.0.0.1:5000`**, membaca alias `champion` dari **`CISO-Demand-Forecaster`**, mengambil run terkait, lalu menimpa **`config/model_metadata.json`**. Snapshot berisi nama/versi/URI model, algoritma, hash dataset DVC, hyperparameter, dan metrik.

Tracking URI dan registry ditetapkan langsung di script; belum tersedia opsi CLI. Server harus sudah memiliki alias dan run tersebut. Script ini tidak mengekspor registry CI atau LK-07. Untuk metadata lifecycle LK-07 gunakan:

```bash
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 export
```

Simpan credential di environment/konfigurasi lokal dan periksa perubahan snapshot sebelum melakukan commit.
