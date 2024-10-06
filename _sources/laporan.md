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

# Simpan data ke file CSV
data.to_csv('data_saham_antam4.csv')
```

Untuk tampilan datanya bisa dilihat di bawah ini:
```{code-cell}
import pandas as pd
import numpy as np

# Membaca data
df = pd.read_csv('https://raw.githubusercontent.com/DeviDuwiSusanti/dataset/main/data_saham_antam4.csv')
pd.options.display.float_format = '{:.0f}'.format
df.head()
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
df.info()
print('Ukuran data ', df.shape)
```
```{code-cell}
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].describe()
```

####  c. Eksplorasi Data
###### Mencari Missing Value
```{code-cell}
df.isnull().sum()
```
###### Mencari Data yang Duplikat
```{code-cell}
duplicates = df[df.duplicated()]
print(duplicates)
```
###### Mengubah kolom Date menjadi Index
Hal ini agar memudahkan akses dan analisis data berdasarkan waktu.
```{code-cell}
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.head()
```
###### Menampilkan Trend Setiap Fitur
```{code-cell}
!pip install seaborn
```
```{code-cell}
import matplotlib.pyplot as plt
import seaborn as sns
for col in df:
    plt.figure(figsize=(7, 3))
    sns.lineplot(data=df, x='Date', y=col)
    plt.title(f'Trend of {col}')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.grid(True)
    plt.xticks(rotation=45) 
    plt.show()

```
###### Mencari Outlier
```{code-cell}
for col in df.columns:
    plt.subplots(figsize=(6, 2))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.grid(True)
    plt.show()
```
Dari boxplot di atas terlihat bahwa fitur 'Volume' memiliki cukup banyak outlier. Sehingga diperlukan adanya penananganan outliernya

```{code-cell}
# Menghitung Z-score untuk kolom Volume
mean_volume = df['Volume'].mean()  # Rata-rata kolom Volume
std_volume = df['Volume'].std()    # Standar deviasi kolom Volume
df['Z_score'] = (df['Volume'] - mean_volume) / std_volume  # Menghitung Z-score

# Menampilkan outlier (Z-score di luar rentang -3 hingga 3)
outliers = df[(df['Z_score'] < -3) | (df['Z_score'] > 3)]
print(f'Jumlah outlier: {outliers.shape[0]}')
print(outliers[['Volume', 'Z_score']])  # Menampilkan kolom yang relevan

# Menghapus outlier dari dataset
df_cleaned = df[(df['Z_score'] >= -3) & (df['Z_score'] <= 3)].copy()  # Buat salinan DataFrame bersih

# Menghapus kolom Z_score setelah pembersihan
df_cleaned.drop(columns=['Z_score'], inplace=True)  # Hapus kolom Z_score

# Menampilkan jumlah data setelah menghapus outlier
print(f'Jumlah data setelah outlier dihapus: {df_cleaned.shape[0]}')

df = df_cleaned
```
###### Rekayasa Fitur
Karena dalam penelitian ini kita akan memprediksi harga Low pada hari berikutnya, maka perlu variabel baru untuk target. Dimana fitur ini dapat membantu kita untuk mengetahui seberapa rendah harga saham bisa turun. Para investor juga bisa menggunakan prediksi ini untuk membeli saham saat harganya rendah, dan meningkatkan peluang mendapatkan keuntungan saat harga saham naik lagi.
```{code-cell}
# Menggeser kolom Adj Close untuk memprediksi keesokan harinya
df['Low Target'] = df['Low'].shift(-1)

# Hapus baris terakhir yang targetnya NaN (karena pergeseran)
df = df[:-1]
df.head()
```
###### Menghitung Korelasi Antar Fitur
```{code-cell}
correlation_matrix = df.corr()

plt.figure(figsize=(7, 3))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()
```
Dari heatmap di atas, bisa dilihat bahwa:
Fitur pembukaan (Open), tertinggi (High), penutupan (Close), dan harga pennutupan yang disesuaikan (Adj Close) mempunyai korelasi yang kuat antara satu sama lain (mendekati 1 atau 1). Hal ini menunjukkan fitur-fitur tersebut saling berkaitan dan bergerak sejalan. Sedangkan fitur 'Volume' mempunyai korelasi rendah dengan fitur lainnya (sekitar 0.15 - 0.17) yang menunjukkan bahwa perubahan volume tidak berpengaruh langsung dengan perubahan harga. Sehingga fitur volume tidak perlu digunakan untuk analisis prediksi pada projek ini.

```{code-cell}
df = df.drop(columns=['Volume'])
df.head()
```
Dari dataset yanng sudah melalui beberapa proses agar siap digunakan, terlihat bahwa fitur input terdiri dari fitur Open, High, Low, Close, Adj Close di hari ini, dan fitur Target Low atau prediksi harga Low besok hari sebagai fitur output.
<!-- ###### Seleksi Fitur
Fitur yang ingin diprediksi adalah fitur Low dimana ini dapat membantu kita untuk mengetahui seberapa rendah harga saham bisa turun. Para investor juga bisa menggunakan prediksi ini untuk membeli saham saat harganya rendah, dan meningkatkan peluang mendapatkan keuntungan saat harga saham naik lagi.
```{code-cell}
# SELEKSI FITUR
df = df.drop(['Open', 'High', 'Adj Close', 'Close', 'Volume'], axis=1)

df
``` -->
### Data Preprocessing
<!-- Langkah-langkah pada tahap ini adalah sebagai berikut :
##### a.	Mengecek missing value -->

<!-- ```{code-cell}
df.isnull().sum()
```
Tujuan : memeriksa apakah ada nilai yang hilang (missing values) dalam dataset.
Fungsi : menampilkan jumlah nilai yang hilang untuk setiap kolom, sehingga jika memang terdapat missing values, kita dapat tangani dengan mengisinya atau menghapus baris-baris yang memiliki missing values. -->

<!-- ##### b.	Pemisahan fitur dan target -->
<!-- Tujuan : memisahkan dataset menjadi fitur (X) dan target (y).
Fungsi : fitur (X) merupakan data yang akan digunakan untuk membuat predikski, sedangkan target (y) adalah nilai yang ingin diprediksi.  -->

<!-- #####  c.	 Normalisasi data -->
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