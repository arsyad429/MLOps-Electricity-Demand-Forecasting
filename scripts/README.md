# Scripts

Folder `scripts/` belum berisi executable source code. Entry point pengumpul data yang aktif berada di `src/pipelines/` agar dapat memakai reusable module dari package `src`.

Jalankan dari root repository:

```powershell
python -m src.pipelines.save_train_historical_data
python -m src.pipelines.save_inference_data
```

Dokumentasi lengkap mengenai setup, output, pemilihan rentang tahun, dan preprocessing tersedia di [`../src/pipelines/README.md`](../src/pipelines/README.md).

File di `scripts/__pycache__/`, jika ada secara lokal, hanyalah bytecode Python dan bukan script yang perlu dijalankan atau di-commit.
