---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Laporan Proyek Sains Data
## Pengembangan Model Prediksi Harga Saham PT Aneka Tambang Tbk (ANTAM) Menggunakan Data Historis untuk Mendukung Keputusan Investasi
## Pendahuluan

### Latar Belakang
Saham merupakan tanda kepemilikan dalam suatu perusahaan yang dapat memberikan hak bagi pemegang saham untuk mendapatkan keuntungan dan ikut serta dalam pengambilan keputusan perusahaan. PT Aneka Tambang Tbk (ANTAM) adalah perusahaan sektor tambang yang memproduksi nikel, emas, dan bauksit, serta sudah terdaftar di Bursa Efek Indonesia sejak 1997. 

ANTAM mempunyai visi untuk menjadi korporasi global terkemuka melalui diversifikasi dan integrasi usaha berbasis sumber daya alam. Untuk memenuhi visi ini, ANTAM memiliki misi yaitu memaksimalkan nilai perusahaan bagi pemegang saham dan pemangku kepentingan dengan cara mengelola biaya secara efisien dan meningkatkan produksi.

Namun, harga saham ANTAM kerap mengalami fluktuasi yang disebabkan oleh faktor-faktor eksternal seperti harga komoditas global dan kondisi ekonomi yang dapat mempersulit para investor dalam membuat keputusan investasi yang tepat.

Untuk mengatasi tantangan ini, diperlukan adanya teknologi yang dapat memprediksi pergerakan harga saham di masa depan, seperti machine learning. Hal ini bertujuan untuk meminimalkan risiko dan membantu investor dalam mengambil keputusan yang tepat.

### Tujuan Proyek
Proyek ini bertujuan untuk mengembangkan model prediksi harga saham di PT Aneka Tambang Tbk (ANTAM)  berakurasi tinggi dengan menggunakan data historis. Dengan analisis ini, diharapkan bisa membantu investor dalam mengambil keputusan investasi, serta dapat memberikan wawasan yang dapat membantu ANTAM dalam merumuskan strategi yang lebih baik untuk meningkatkan nilai saham dan mencapai tujuan pertumbuhannya.

### Rumusan Masalah
•	Bagaimana cara untuk mengembangkan sebuah model prediksi harga saham PT Aneka Tambang Tbk (ANTAM) yang akurat dengan menggunakan data historis?

•	Bagaimana hasil prediksi harga saham dapat dimanfaatkan untuk mendukung keputusan investasi yang lebih baik dan membantu ANTAM  merancang strategi agar meningkatkan nilai saham dan pertumbuhan perusahaan?

## METODOLOGI
### Data Understanding
#### a.	Sumber Data
Data yang dipakai pada proyek ini didapat dari website https://finance.yahoo.com/quote/ANTM.JK/history/, yaitu sebuah platform online yang menyediakan data keuangan dan pasar saham secara real-time. Di website tersebut kita bisa menemukan informasi atau data historis harga saham dari PT Aneka Tambang Tbk (ANTM) di Bursa Efek Jakarta. Di dalam proyek ini, digunakan data histori dari tanggal 09-09-2015 sampai 10-09-2024 dalam bentuk dokumen csv. Dalam pengambilan data dari Yahoo Finance, kita bisa menggunakan google colab untuk mendownload data yang kita butuhkan. Berikut adalah code yang bisa digunakan :

```python
import yfinance as yf
import pandas as pd

# Unduh data saham ANTAM dari Yahoo Finance
data = yf.download("ANTM.JK", start="2015-09-09", end="2024-09-10")

# Tampilkan beberapa baris pertama dari data untuk verifikasi
print(data.head())

# Simpan data ke file CSV
data.to_csv('data_saham_antam4.csv')

print("Data telah disimpan ke file data_saham_antam4.csv")
```
#### b.	Deskripsi Data Set
Data set ini terdiri dari 8 fitur atau kolom, dan 2230 record atau baris.
Atribut-atribut data set :
- Date		: tanggal data harga saham, biasanya memiliki format YYYY-MM-DD.
- Open		: harga pembukaan saham pada tanggal tersebut.
- High		: harga tertinggi yang dicapai pada tanggal tersebut.
- Low		: harga terendah saham pada tanggal tersebut.
- Close		: harga penutupan saham pada tanggal tersebut.
- Adj Close	: harga penutupan yang sudah disesuaikan dengan pembagian saham,     
  		       dividen, dan corporate actions lainnya.
- Volume	: jumlah saham yang diperdagangkan pada tanggal tersebut.
- Adj Close Target : harga target yang akan diprediksi untuk besok hari

```{code-cell}
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data
df = pd.read_csv('https://raw.githubusercontent.com/DeviDuwiSusanti/dataset/main/data_saham_antam4.csv')

# Menggeser kolom Adj Close untuk memprediksi keesokan harinya
df['Adj Close Target'] = df['Adj Close'].shift(-1)

# Hapus baris terakhir yang targetnya NaN (karena pergeseran)
df = df[:-1]
print(df)

df.info()
print('Ukuran data ', df.shape)
```

```{code-cell}
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Adj Close Target']].describe()
```

### Data Preprocessing
Langkah-langkah pada tahap ini adalah sebagai berikut :
##### a.	Mengecek missing value

```{code-cell}
df.isnull().sum()
```
Tujuan : memeriksa apakah ada nilai yang hilang (missing values) dalam dataset.
Fungsi : menampilkan jumlah nilai yang hilang untuk setiap kolom, sehingga jika memang terdapat missing values, kita dapat tangani dengan mengisinya atau menghapus baris-baris yang memiliki missing values.

##### b.	Pemisahan fitur dan target
```{code-cell}
X = df[['Open', 'High', 'Low', 'Close', 'Adj Close']]
y = df['Adj Close Target']
```
Tujuan : memisahkan dataset menjadi fitur (X) dan target (y).
Fungsi : fitur (X) merupakan data yang akan digunakan untuk membuat predikski, sedangkan target (y) adalah nilai yang ingin diprediksi.

#####  c.	 Normalisasi data
<!-- ```{code-cell}
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
Tujuan : menormalkan data fitur ke dalam rentang [0,1].
Fungsi : menggunakan MinMaxScaler untuk memastikan bahwa semua fitur berada dalam rentang yang sama supaya skla data konsisten, agar algoritma berfungsi dengan baik. -->

### Modelling
Menjelaskan proses pembuatan model berdasarkan data yang sudah kita proses
##### a.	Pembagian Data
Data dibagi menjadi dua, yaitu data pelatihan untuk melatih model dan data pengujian untuk mengecek seberapa baik model bekerja.

##### b.	Pembangunan Model
Pemilihan algoritma seperti SVM, Naïve Bayes, atau yang lainnya dan melatih model menggunakan data pelatihan untuk mengenali pola dalam data.

##### c.	Pengujian Model
Model diuji dengan data pengujian untuk melihat seberapa akurat prediksi yang dihasilkan.

### Evaluation
Di sini adalah tempat kita untuk mengukur kinerja model menggunakan metrik yang relevan seperi akurasi contohnya. Hal ini menentukan apakah model yang kita capai sudah memadai untuk digunakan dalam aplikasi nyata atau tidak.

## Kesimpulan

## Referensi