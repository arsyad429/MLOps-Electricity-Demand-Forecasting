# Monitoring

Folder ini disiapkan untuk pemantauan performa dan drift model demand CISO **t+2 jam**. Saat ini belum berisi modul monitoring, penghitungan drift, alert, atau pemicu retraining.

## Bukti evaluasi yang sudah tersedia

| Sumber | Informasi |
| --- | --- |
| MLflow training runs | Validation/test MAE, RMSE, WAPE (%), parameter, target/horizon, dan hash dataset |
| `config/model_metadata.json` | Snapshot registry training manual, diperbarui melalui script export |
| Metadata LK-07 yang dilacak DVC | Versi, stage, alias, parameter, metrik, dan lineage lifecycle |
| Artifact Actions `mlops-evidence-<run_id>` | Candidate, evaluation report, registration report jika tersedia, dan pointer fitur |

Evaluation gate saat ini memakai validation RMSE <= 1200 dan WAPE <= 3,0% dari `config/ci_thresholds.yaml`. Kandidat yang lolos masuk registry CI pada Staging. Gate ini mengevaluasi hasil training; belum memantau akurasi prediksi live atau perubahan distribusi data produksi.

## Rencana integrasi

Monitoring live memerlukan pencatatan prediksi beserta timestamp, horizon, versi model, dan fitur input, lalu pencocokan dengan aktual setelah dua jam. Setelah data tersebut tersedia, pemantauan dapat menghitung metrik per jendela waktu, drift fitur/target, dan keterlambatan/kualitas data.

Retraining yang tersedia sekarang dipicu perubahan kode/push ke `main` atau `workflow_dispatch` pada ref `main`. Belum ada jadwal retraining, trigger drift, promosi Production otomatis, atau dashboard monitoring. Detail perilaku saat ini tersedia pada [README models](../models/README.md), [README scripts](../../scripts/README.md), dan [README utama](../../README.md#github-actions-ci-dan-continuous-training).
