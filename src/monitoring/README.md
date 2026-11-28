# Monitoring

Evaluasi produksi CISO **t+2 jam** mencocokkan prediksi yang tersimpan dengan demand EIA pada timestamp target UTC yang sama. SQLite menyimpan aktual, sumber, nomor revisi, dan waktu evaluasi per ID prediksi.

## Evaluasi prediksi live

Pipeline inferensi mencoba mengevaluasi prediksi lama pada setiap siklus. Target yang sudah lewat dan belum memiliki aktual berstatus `pending_actual`. Setelah aktual tersedia, evaluasi menghitung error dari prediksi awal. Jika EIA merevisi aktual, siklus berikutnya memperbarui hasil evaluasi dan riwayat revisi tetap tersedia di `observation_revisions`.

```powershell
python -m src.pipelines.evaluate_predictions --window-hours 168
python -m src.storage.prediction_store evaluations --limit 5
```

Perintah evaluasi memproses semua pasangan yang siap dan menampilkan ringkasan sesuai jendela target yang dipilih. Pakai `--summary-only` untuk membaca hasil yang tersimpan. Laporan memisahkan MAE dan RMSE dalam MWh, serta WAPE dalam persen, berdasarkan nama dan versi model. WAPE bernilai `null` jika total aktual nol. `evaluations_updated` menghitung evaluasi baru dan evaluasi yang diperbarui.

SQLite bermigrasi otomatis ke skema 2. Untuk volume Docker, gunakan image worker terbaru agar siklus inferensi dan perintah evaluasi memakai versi skema yang sama. Lihat [petunjuk worker Docker](../pipelines/README.md#hourly-docker-worker).

## Bukti evaluasi yang sudah tersedia

| Sumber | Informasi |
| --- | --- |
| MLflow training runs | Validation/test MAE, RMSE, WAPE (%), parameter, target/horizon, dan hash dataset |
| `config/model_metadata.json` | Snapshot registry training manual, diperbarui melalui script export |
| Metadata LK-07 yang dilacak DVC | Versi, stage, alias, parameter, metrik, dan lineage lifecycle |
| Artifact Actions `mlops-evidence-<run_id>` | Candidate, evaluation report, registration report jika tersedia, dan pointer fitur |

Evaluation gate untuk kandidat training memakai validation RMSE <= 1200 dan WAPE <= 3,0% dari `config/ci_thresholds.yaml`. Kandidat yang lolos masuk registry CI pada Staging.

## Rencana integrasi

Retraining yang tersedia sekarang dipicu perubahan kode/push ke `main` atau `workflow_dispatch` pada ref `main`. Belum ada jadwal retraining, trigger drift, promosi Production otomatis, atau dashboard monitoring. Detail perilaku saat ini tersedia pada [README models](../models/README.md), [README scripts](../../scripts/README.md), dan [README utama](../../README.md#github-actions-ci-dan-continuous-training).
