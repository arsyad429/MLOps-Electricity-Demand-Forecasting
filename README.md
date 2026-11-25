# Electricity Demand Forecasting with MLOps

Sistem forecasting demand listrik **California ISO (CISO) untuk t+2 jam**, menggunakan data hourly EIA, XGBoost/LightGBM, Optuna, MLflow, DVC, dan GitHub Actions. Timestamp dataset menggunakan UTC; fitur kalender menggunakan zona waktu `America/Los_Angeles`.

## Status implementasi

| Komponen | Kondisi saat ini |
| --- | --- |
| Ingestion | Client EIA API v2, pembaruan historis incremental, dan pengambilan record terbaru |
| Preprocessing | Normalisasi waktu/nilai dan target `target_demand_2h` berdasarkan pencocokan timestamp |
| Feature engineering | Kalender California, lag 1/2/24 baris, dan rolling mean 3/24 baris |
| Training | Tuning XGBoost dan LightGBM dengan Optuna; split kronologis dan pemilihan berdasarkan validation RMSE |
| Tracking dan registry | MLflow runs, artifacts, signature, dataset lineage, alias, dan lifecycle LK-07 |
| Versioning data | Pointer DVC untuk dataset raw, fitur training, dan metadata LK-07 |
| CI dan continuous training | Automated tests, retraining pada `main`, evaluation gate, dan registrasi kandidat ke Staging |
| Backend, frontend, monitoring | Folder rencana; aplikasi serving, dashboard, dan drift detection belum tersedia |

Inferensi yang sudah tersedia adalah smoke test model registry menggunakan sampel historis. Pengumpulan satu record terbaru belum membentuk pipeline prediksi live karena fitur model memerlukan riwayat demand.

## Struktur dan dokumentasi

```text
MLOps-Electricity-Demand-Forecasting/
|-- .devcontainer/           # Python 3.11 dan konfigurasi Codespaces
|-- .github/workflows/       # CI dan automated retraining
|-- artifacts/lk07/          # Pointer DVC metadata lifecycle
|-- backend/                # Rencana FastAPI serving
|-- config/                 # YAML, threshold CI, snapshot metadata model
|-- data/                   # raw, interim, processed, external
|-- docs/                   # Placeholder dokumentasi tambahan
|-- frontend/               # Rencana dashboard React + Vite
|-- models/                 # Placeholder artifacts model lokal
|-- notebooks/              # Eksplorasi akses API EIA
|-- scripts/                # Evaluation gate dan export metadata
|-- src/
|   |-- data/               # Client API dan preprocessing
|   |-- features/           # Transformasi fitur
|   |-- models/             # Training dan lifecycle registry
|   |-- monitoring/         # Rencana monitoring
|   `-- pipelines/          # Entry point persiapan data
`-- tests/                  # Pengujian metrik, split, fitur, dan model
```

Dokumentasi komponen: [data](src/data/README.md), [pipelines](src/pipelines/README.md), [features](src/features/README.md), [models](src/models/README.md), [scripts](scripts/README.md), [tests](tests/README.md), [notebooks](notebooks/README.md), [backend](backend/README.md), [frontend](frontend/README.md), dan [monitoring](src/monitoring/README.md).

## Persiapan environment

Gunakan Python **3.11+**, dengan Python 3.11 sebagai versi CI dan devcontainer. Jalankan perintah dari root repository.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Isi `EIA_API_KEY` di `.env` untuk ingestion. `DATABASE_URL` masih berupa placeholder dan belum dipakai oleh aplikasi backend. Untuk remote DVC DagsHub, pasang tambahan `python -m pip install "dvc[s3]"`.

Dependency belum dikunci versinya. Training memakai API MLflow seperti `log_model(name=...)`; gunakan environment MLflow yang mendukung API tersebut. Codespaces memasang dependency Python otomatis dan meneruskan port 5000, 8000, serta 5173; service terkait tetap perlu dijalankan sendiri.

## Memulihkan dataset dengan DVC

Git menyimpan pointer `.dvc`, sedangkan isi dataset/artifact berada di cache atau remote DVC. Repository sudah diinisialisasi dengan dua remote:

| Remote | Lokasi | Penggunaan |
| --- | --- | --- |
| `localstorage` (default) | `E:/DVC-Storage/MLOps-Electricity-Demand-Forecasting` | Storage lokal Windows; memerlukan drive/path tersebut |
| `dagshub` | `s3://dvc`, endpoint DagsHub repository ini | Pemulihan dan penyimpanan dataset oleh workflow CI |

Periksa konfigurasi dengan `dvc remote list`. Pada mesin lain atau Linux, pilih `-r dagshub` secara eksplisit. Setelah menyiapkan credential lokal dari token DagsHub:

```powershell
dvc remote modify --local dagshub access_key_id $env:DAGSHUB_TOKEN
dvc remote modify --local dagshub secret_access_key $env:DAGSHUB_TOKEN
dvc pull -r dagshub data/raw/historical/ciso_historical.csv.dvc
```

Pada Bash, gunakan `"$DAGSHUB_TOKEN"` untuk nilai credential. Opsi `--local` menyimpan credential di `.dvc/config.local`, yang diabaikan Git. Pengunduhan memerlukan akses dan object yang telah diunggah ke remote; keberadaan pointer tidak menjamin object tersedia di setiap remote.

Untuk memakai fitur training yang sudah dilacak, jalankan:

```bash
dvc pull -r dagshub data/processed/historical/ciso_train_features.csv.dvc
```

## Alur data dan training lokal

### 1. Pembaruan data historis (opsional)

```bash
python -m src.pipelines.save_train_historical_data
```

Pipeline memerlukan `data/raw/historical/ciso_historical.csv` yang sudah ada. Data baru diambil mulai satu jam setelah timestamp terakhir hingga jam UTC sekarang dikurangi satu jam, dalam chunk maksimal 24 jam. Request tertentu diulang maksimal tiga percobaan. Hasil digabung, dideduplikasi berdasarkan `period`, dan menimpa file yang sama. Entry point ini tidak menyediakan argumen rentang tahun atau bootstrap dataset kosong.

### 2. Preprocessing dan feature engineering

```bash
python -m src.pipelines.preprocess_train_data
python -m src.pipelines.build_train_features
dvc add data/processed/historical/ciso_train_features.csv
```

Output berturut-turut adalah `data/interim/historical/processed_ciso_historical.csv` dan `data/processed/historical/ciso_train_features.csv`. Target berasal dari demand pada timestamp **tepat dua jam setelah `period`**. Baris tanpa target atau fitur lengkap dibuang oleh pipeline fitur.

Training membaca hash dari pointer `.dvc` sebagai lineage. Setelah mengubah CSV, perbarui pointer sebelum training agar hash yang dicatat sesuai dataset. Jika raw juga berubah, jalankan `dvc add data/raw/historical/ciso_historical.csv`. Unggah data ke remote yang dipakai tim, kemudian commit pointer dan perubahan terkait ke Git; CSV lengkap dan cache tidak perlu dimasukkan ke Git.

### 3. Jalankan MLflow

Pada terminal terpisah dengan environment aktif:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Database dan artifacts lokal diabaikan Git. Server baru tidak otomatis memiliki runs atau versi registry dari mesin lain.

### 4. Training dan perbandingan model

```bash
python -m src.models.train --trials 10 --tracking-uri http://127.0.0.1:5000
```

`--trials` berlaku **per algoritma**; default CLI adalah 50. Training memvalidasi dataset, membagi data sekitar 70/15/15 secara kronologis, dan membuang label yang melewati batas split untuk horizon dua jam. Optuna meminimalkan validation RMSE; kandidat terbaik dari dua algoritma dipilih berdasarkan metrik yang sama. Test set dievaluasi setelah pemilihan kandidat untuk pelaporan MAE, RMSE, dan WAPE (%).

Mode manual mendaftarkan kedua model ke `CISO-Demand-Forecaster` lalu mengatur alias `champion`. Mode `--ci` menyimpan artifacts model dan `ci_output/candidate.json` tanpa registrasi atau perubahan alias saat training. Detail fitur, validasi, dan registry tersedia di [README models](src/models/README.md).

## Data inference dan model lifecycle

```bash
python -m src.pipelines.save_inference_data
python -c "from src.pipelines.preprocess_inference_data import preprocess_inference_data; print(preprocess_inference_data('data/raw/inference/ciso_inference.csv'))"
```

Ingestion meminta jendela tiga jam terakhir hingga jam UTC sekarang dan memilih satu record terbaru. Respons kosong akibat keterlambatan publikasi EIA belum ditangani secara khusus. Preprocessing inference menghasilkan `data/interim/inference/processed_ciso_inference.csv` tanpa target. Belum ada entry point untuk menyusun fitur inference dari riwayat lalu menjalankan prediksi live.

Tiga registry mempunyai fungsi berbeda:

| Registry | Pengelola | Alias/hasil |
| --- | --- | --- |
| `CISO-Demand-Forecaster` | Training manual | `champion`, pemenang validation RMSE |
| `CISO-Demand-Forecaster-LK07` | CLI lifecycle LK-07 | Promosi manual None -> Staging -> Production; alias `staging`/`champion` |
| `CISO-Demand-Forecaster-CI` | Evaluation gate CI | Kandidat lolos evaluasi masuk Staging dengan alias `staging` |

Snapshot [`config/model_metadata.json`](config/model_metadata.json) mencatat XGBoost v3 dari registry manual. Metadata LK-07 dilacak oleh [`artifacts/lk07/model_lineage.json.dvc`](artifacts/lk07/model_lineage.json.dvc); snapshot lokal yang diperiksa mencatat v2 sebagai Production dan `champion`, dengan `n_estimators` 600 -> 550. MAE/WAPE validasi membaik (720,93 -> 715,19; 2,736% -> 2,714%), sementara RMSE sedikit naik (1.028,97 -> 1.030,16). Snapshot tersebut merupakan bukti eksperimen, bukan pemeriksaan status registry live.

Pada MLflow server yang memiliki registry LK-07 dan dataset terkait:

```bash
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 status
python -m src.models.registry --tracking-uri http://127.0.0.1:5000 verify
```

`verify` membandingkan prediksi dari alias `champion` dan stage Production menggunakan baris terakhir dataset historis. Langkah registrasi dan promosi dijelaskan di [README models](src/models/README.md).

## GitHub Actions: CI dan continuous training

Workflow [`ci.yml`](.github/workflows/ci.yml) berjalan pada push dan pull request untuk memasang dependency serta memverifikasi environment Python 3.11.

Workflow [`mlops-automation.yaml`](.github/workflows/mlops-automation.yaml) berjalan pada push/PR ke `main` dan `workflow_dispatch`:

1. Menjalankan `python -m pytest tests/ -v`.
2. Setelah tests lolos, training berjalan untuk event selain PR pada ref `main`.
3. Mengambil dataset raw dari remote DagsHub, menjalankan preprocessing dan feature engineering, lalu memperbarui pointer fitur dengan `dvc add`.
4. Menjalankan Optuna sebanyak tiga trial per algoritma dalam mode `--ci`.
5. Menguji kandidat dengan threshold [`config/ci_thresholds.yaml`](config/ci_thresholds.yaml): **validation RMSE <= 1200** dan **validation WAPE <= 3,0%**. Nilai harus finite; metrik test bukan dasar pemilihan atau gate.
6. Jika gate lolos, mengunggah dataset fitur ke DagsHub lalu mendaftarkan kandidat ke registry CI pada Staging.
7. Mengunggah evidence `ci_output/` dan pointer fitur sebagai artifact `mlops-evidence-<run_id>`, termasuk pada kegagalan jika file tersedia.

Repository settings perlu menyediakan:

| Jenis | Nama | Kegunaan |
| --- | --- | --- |
| Actions secret | `DAGSHUB_TOKEN` | Credential DVC dan password MLflow |
| Actions variable | `MLFLOW_TRACKING_URI` | Endpoint tracking MLflow |
| Actions variable | `DAGSHUB_USERNAME` | Username MLflow |

Job training memakai repository secrets/variables tanpa deklarasi GitHub Environment. Workflow ini memakai data raw versi DVC, tidak mengambil data EIA baru, tidak melakukan promosi Production otomatis, dan tidak melakukan commit pointer hasil training ke repository. Push dokumentasi ke `main` juga memicu workflow karena belum ada filter path.

## Pengujian

```bash
python -m pytest tests/ -v
```

Suite saat ini mencakup lima kasus: metrik, batas chronological split, nilai fitur, dan fit/predict untuk dua algoritma. Tests memakai data sintetis dan tidak memerlukan EIA, dataset DVC, atau server MLflow aktif. Cakupan belum mencakup integrasi remote dan seluruh alur CI; lihat [README tests](tests/README.md).

## Konfigurasi dan batas implementasi

`config/ci_thresholds.yaml` dibaca langsung oleh evaluation gate. Sebaliknya, `config/data.yaml`, `config/model.yaml`, dan `config/pipeline.yaml` masih bersifat deklaratif dan belum menjadi sumber parameter runtime pipeline/training. `config/data.yaml` masih mencatat horizon satu jam dan daftar lag di `config/model.yaml` berbeda dari implementasi. Untuk perilaku aktual, gunakan `target_demand_2h`, fungsi `add_features`, dan `FEATURE_COLUMNS` di kode training.

Lag dan rolling dihitung berdasarkan posisi baris; data harus terurut dengan interval hourly. Training menyaring baris dengan konteks 24 jam yang tidak lengkap, tetapi belum ada validasi menyeluruh atau perbaikan gap pada ingestion. API serving, penyusunan fitur inference live, dashboard, drift detection, dan retraining berbasis jadwal/drift masih menjadi pekerjaan berikutnya.

Simpan credential di `.env`, environment variables, Actions secrets, atau konfigurasi DVC lokal. Jangan commit token, database MLflow, cache, atau artifacts model lokal.

## Docker Compose 

Layanan: `mlflow-server` (SQLite metadata + model artifacts pada named volumes) dan
`api-service` (FastAPI, prediksi CISO t+2h). Keduanya di network `ciso-network`.

1. Pastikan Docker Desktop menjalankan Linux containers.
2. Dari root repo di Windows PowerShell, jalankan:

   ```powershell
   Copy-Item .env.compose.example .env.compose
   notepad .env.compose
   docker compose config --quiet
   docker compose up -d --build
   docker compose ps
   ```

3. Isi DAGSHUB_TOKEN pada `.env.compose` dengan token aktif, jangan commit.
4. MLflow UI: http://127.0.0.1:5001 (container internal: `http://mlflow-server:5000`).
5. API docs: http://127.0.0.1:8000/docs.
6. Verifikasi komunikasi internal:

   ```powershell
   docker compose exec api-service python -c "import requests; r=requests.get('http://mlflow-server:5000/health', timeout=5); print(r.status_code, r.text)"
   Invoke-RestMethod http://127.0.0.1:8000/mlflow-check | ConvertTo-Json
   ```

7. Contoh prediksi dengan engineered features (bukan data live):

   ```powershell
   $body = @{value=25000;lag_1h=24900;lag_2h=24800;lag_24h=24300;rolling_mean_3h=24900;rolling_mean_24h=24400;hour=13;day_of_week=1;month=6;is_weekend=0} | ConvertTo-Json
   Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
   ```

8. Stop tanpa menghapus data: `docker compose down`.
   Jangan gunakan `docker compose down -v` kecuali ingin menghapus seluruh database dan artifacts.

Saat pertama startup, `backend/bootstrap_model.py` mengunduh model dari
DagsHub `CISO-Demand-Forecaster-CI@staging`, mencatatnya ke MLflow
lokal bernama `CISO-Demand-Forecaster-Compose`, lalu memberi alias `champion`.
Saat restart, import tidak diulang jika alias lokal sudah ada.
Perubahan staging sumber tidak langsung memutakhirkan model lokal.
Untuk impor ulang sumber secara sengaja:

```powershell
docker compose exec api-service python -m backend.bootstrap_model --force
docker compose restart api-service
```

Contoh inferensi menggunakan fitur hasil rekayasa dan bukan API prediksi EIA live.
Setup ini untuk demonstrasi lokal, bukan production publik.


## MLflow Model Serving & Horizontal Scaling

Tahap ini bertujuan untuk melakukan model serving menggunakan MLflow, membangun Docker image dari model yang telah dilatih, serta menerapkan horizontal scaling dengan tiga replika container melalui Docker Compose.

### 1. Persiapan Model Production

Pastikan model `CISO-Demand-Forecaster-Compose` sudah terdaftar pada MLflow Model Registry dengan status **Production**.

Jalankan layanan MLflow dan FastAPI dari LK-09:

```powershell
docker compose up -d mlflow-server api-service
```

Akses MLflow UI melalui:

http://127.0.0.1:5001

### 2. Model Serving Lokal

Aktifkan Python virtual environment proyek dan atur MLflow Tracking URI:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
```

Jalankan model serving menggunakan MLflow:

```powershell
mlflow models serve `
  --model-uri "models:/CISO-Demand-Forecaster-Compose/Production" `
  --host 127.0.0.1 `
  --port 1234 `
  --env-manager local
```

Model akan tersedia melalui endpoint:

```text
POST http://127.0.0.1:1234/invocations
```

Endpoint `/invocations` menerima input fitur dalam format JSON dan menghasilkan prediksi demand listrik untuk dua jam berikutnya.

### 3. Pengujian API

Contoh payload yang digunakan untuk pengujian:

```json
{
  "dataframe_records": [
    {
      "value": 25000.0,
      "lag_1h": 24900.0,
      "lag_2h": 24800.0,
      "lag_24h": 24300.0,
      "rolling_mean_3h": 24900.0,
      "rolling_mean_24h": 24400.0,
      "hour": 13,
      "day_of_week": 1,
      "month": 6,
      "is_weekend": 0
    }
  ]
}
```

Simpan payload tersebut pada `tests/payloads/lk10_prediction.json`.

Pada terminal PowerShell lain, jalankan:

```powershell
curl.exe -X POST "http://127.0.0.1:1234/invocations" `
  -H "Content-Type: application/json" `
  --data-binary "@tests/payloads/lk10_prediction.json"
```

Respons API berisi nilai prediksi dalam field `predictions`.

### 4. Kontainerisasi Model

Setelah model serving lokal berhasil, bangun Docker image menggunakan MLflow.

Pastikan MLflow Tracking Server masih berjalan, kemudian jalankan:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"

mlflow models build-docker `
  --model-uri "models:/CISO-Demand-Forecaster-Compose/Production" `
  --name "ciso-mlflow-serving:lk10"
```

Perintah tersebut menghasilkan Docker image yang membungkus model beserta dependensi dan server inferensinya.

Verifikasi image:

```powershell
docker images ciso-mlflow-serving
```

Image `ciso-mlflow-serving:lk10` digunakan oleh service `model-serving` dalam `docker-compose.yaml`.

### 5. Horizontal Scaling dengan Docker Compose

Service `model-serving` pada `docker-compose.yaml` dikonfigurasi menggunakan:

```yaml
model-serving:
  image: ciso-mlflow-serving:lk10

  ports:
    - "127.0.0.1::8080"

  networks:
    - ciso-network

  deploy:
    replicas: 3
```

Ketiga replika menggunakan Docker image dan model yang sama, tetapi berjalan dalam container terpisah.

Validasi konfigurasi:

```powershell
docker compose config --quiet
```

Jalankan tiga replika:

```powershell
docker compose up -d --scale model-serving=3
```

Verifikasi status container:

```powershell
docker compose ps
```

Ketiga container `model-serving` diharapkan berstatus `Up` dan `healthy`.

### 6. Mengakses Endpoint Setiap Replika

Setiap replika menggunakan port internal `8080`, tetapi Docker memberikan port host yang berbeda agar tidak terjadi konflik.

Tampilkan port masing-masing replika:

```powershell
docker compose port --index 1 model-serving 8080
docker compose port --index 2 model-serving 8080
docker compose port --index 3 model-serving 8080
```

Contoh hasil:

```text
127.0.0.1:55001
127.0.0.1:55002
127.0.0.1:55003
```

Nomor port host ditentukan secara dinamis oleh Docker dan dapat berbeda pada setiap deployment.

Untuk menguji prediksi pada ketiga replika:

```powershell
$body = Get-Content "tests\payloads\lk10_prediction.json" -Raw

for ($i = 1; $i -le 3; $i++) {
    $address = (
        docker compose port --index $i model-serving 8080
    ).Trim()

    Write-Host "Testing replica $i at $address"

    Invoke-RestMethod `
      -Uri "http://$address/invocations" `
      -Method Post `
      -ContentType "application/json" `
      -Body $body
}
```

Setiap replika seharusnya menghasilkan prediksi yang sama atau sangat mendekati karena menggunakan model dan input yang identik.

### 7. Dynamic Scaling

Jumlah replika dapat ditambah atau dikurangi tanpa melakukan retraining maupun membangun ulang Docker image.

Menambah menjadi lima replika:

```powershell
docker compose up -d --scale model-serving=5
```

Memverifikasi jumlah container:

```powershell
docker compose ps
```

Mengembalikan menjadi tiga replika:

```powershell
docker compose up -d --scale model-serving=3
```

### 8. Catatan Implementasi

- Horizontal scaling dilakukan dengan menjalankan beberapa instance dari Docker image yang sama.
- Replikasi meningkatkan potensi kapasitas pemrosesan request, bukan akurasi model.
- Setiap replika memiliki proses dan penggunaan memori terpisah.
- Saat ini setiap replika diakses melalui port host masing-masing. Load balancing otomatis melalui satu endpoint belum diimplementasikan.
- Model sudah tertanam dalam Docker image. Perubahan model Production memerlukan rebuild image dan redeployment.
- Persistent volumes MLflow dari LK-09 tetap digunakan untuk menyimpan metadata dan artifacts model.
- Jangan menjalankan `docker compose down -v` jika ingin mempertahankan data MLflow.

