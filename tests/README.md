# Automated tests

[`test_training.py`](test_training.py) berisi pengujian dasar untuk training demand CISO t+2 jam. Suite menggunakan data sintetis dan berjalan tanpa API key EIA, dataset DVC, atau server MLflow aktif. Dependency model dan MLflow tetap harus terpasang karena diimpor oleh modul training.

## Menjalankan

Dari root repository dengan environment Python aktif:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

Untuk satu kelompok pengujian:

```bash
python -m pytest tests/test_training.py -k feature_engineering -v
python -m pytest tests/test_training.py -k model_training -v
```

## Cakupan saat ini

| Test | Pemeriksaan |
| --- | --- |
| `test_metrics` | Nilai MAE, RMSE, dan WAPE pada pasangan actual/predicted sederhana |
| `test_chronological_split` | Split tidak kosong dan label t+2 train/validation tidak mencapai batas split berikutnya |
| `test_feature_engineering` | Nilai lag 1/2/24 dan rolling mean tiga baris pada seri hourly sintetis |
| `test_model_training[xgboost]` | Model dapat di-fit dan menghasilkan 50 prediksi finite |
| `test_model_training[lightgbm]` | Pemeriksaan fit/predict yang sama untuk LightGBM |

Terdapat lima kasus setelah parametrization. Tests belum mencakup request/retry EIA, target timestamp preprocessing, loader dengan gap, seluruh fitur kalender, tuning Optuna, evaluation gate, registrasi/promosi MLflow, akses DVC remote, atau serving.

## Hubungan dengan CI

Job `automated-testing` pada [workflow MLOps](../.github/workflows/mlops-automation.yaml) menjalankan `python -m pytest tests/ -v` pada Python 3.11. Job training membutuhkan tests lolos dan hanya berjalan untuk event selain PR pada ref `main`. Workflow [CI environment](../.github/workflows/ci.yml) hanya memasang dependency dan memeriksa environment Python.

Kelulusan suite dasar tidak membuktikan kredensial remote, threshold performa kandidat, atau seluruh pipeline retraining berhasil; hasilnya terlihat pada job training dan evidence CI.
