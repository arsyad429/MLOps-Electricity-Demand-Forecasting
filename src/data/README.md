# Data Modules

Folder ini berisi fungsi reusable untuk mengambil dan membersihkan data demand CISO.

## Isi Modul

### `ingest_data.py`

- `get_ciso_EIA_data(start_date=None, end_date=None, train=True)` mengambil data hourly demand dari EIA API v2.
- `save_raw_data(df, prefix)` menyimpan DataFrame sebagai CSV bertimestamp UTC di bawah `data/raw/`.

Dalam mode training, `start_date` dan `end_date` wajib memakai format yang diterima EIA, misalnya `2024-01-01T00`. Dalam mode inference (`train=False`), request dimulai dari jam UTC saat fungsi dipanggil.

Contoh mengambil satu rentang data tanpa menyimpannya:

```python
import pandas as pd

from src.data.ingest_data import get_ciso_EIA_data

result = get_ciso_EIA_data(
    start_date="2024-01-01T00",
    end_date="2024-01-07T23",
)
df = pd.DataFrame(result["response"]["data"])
```

Fungsi membaca `EIA_API_KEY` dari environment atau file `.env` di root project. Request menggunakan timeout 30 detik dan meneruskan kegagalan HTTP sebagai exception.

### `preprocess.py`

`formating_ciso_data(df, train=True)` melakukan preprocessing berikut:

1. mengubah `period` menjadi datetime UTC;
2. mengubah `value` menjadi numerik;
3. membuang duplikat dan nilai kosong;
4. mengurutkan baris berdasarkan `period`; dan
5. menambahkan `target_demand_1h` untuk data training.

Nama fungsi `formating_ciso_data` mengikuti implementasi saat ini.

## Menjalankan Pipeline

Modul di folder ini merupakan library dan bukan entry point CLI. Untuk mengunduh serta menyimpan data, jalankan pipeline dari root repository:

```powershell
python -m src.pipelines.save_train_historical_data
python -m src.pipelines.save_inference_data
```

Lihat [`../pipelines/README.md`](../pipelines/README.md) untuk detail command, output, dan contoh preprocessing.
