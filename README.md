## MAGPIE Framework (Multiclass Edition)
Sebuah sistem klasifikasi varian genetik berbasis LightGBM dengan integrasi OpenFE dan imputasi data IterativeImputer. Versi ini telah dioptimalkan untuk berjalan secara efisien pada lingkungan memori terbatas dan mendukung klasifikasi multikelas (Benign, VUS, Pathogenic).

### Basic Requirements 
1. Python 3.9+
2. Paket Python yang terdaftar dalam `requirements.txt`.
3. Library SpliceAI.

Sistem ini didesain agar mudah diatur. Gunakan `bash jumpstart.sh` untuk menyiapkan lingkungan Miniconda dan dependensi MAGPIE secara otomatis di Kaggle atau Linux.

### Usage
Sistem MAGPIE membaca data input varian dalam format CSV. Pastikan lima kolom awal tabel bernama `Chr`, `Start`, `End`, `Ref`, dan `Alt`. [Contoh File](data/datasets/test.csv)

|  Chr  | Start |  End  |  Ref  |  Alt  |  ...  |
| ----- | ----- | ----- | ----- | ----- | ----- |

#### Use Pretrained Model to Predict Variants
Sistem membaca dan mengeksekusi model yang telah Anda latih menggunakan data set multikelas sebelumnya.

**Annotated Variants (Jika file input sudah dianotasi)**
Jalankan perintah ini:
`source magpie.sh --mode pred --test_file data/datasets/test.csv --file_state annotated --visualization`

**Unannotated Variants (Jika file input mentah)**
Jalankan perintah ini:
`source magpie.sh --mode pred --test_file data/datasets/test.csv --file_state unannotated --visualization`

Hasil klasifikasi dan metrik performa akan diekspor ke dalam direktori `data/result` dan `data/output/visualization/`.

#### Train Model From Scratch (De Novo)
Versi ini telah memangkas ketergantungan MATLAB dan OMIM. Sistem sepenuhnya independen menggunakan Python murni dan basis data publik.

1.  Daftarkan diri Kamu untuk mendapatkan perangkat lunak ANNOVAR melalui situs web resmi mereka. Ekstrak dan tempatkan skrip eksekusi perl (`table_annovar.pl`, dll.) di dalam direktori `annovar/`.
2.  Jalankan perintah `bash download.sh` untuk mengunduh basis data klinis hg38, SpliceAI genom referensi, ChromHMM, dan GenCC (pengganti OMIM) secara otomatis.
3.  Jalankan perintah pelatihan model menggunakan skrip bash:
    `source magpie.sh --mode train --input_file data/datasets/denovo.csv`

Model hasil pelatihan akan disimpan ke dalam direktori `data/result/MAGPIE.model`.
