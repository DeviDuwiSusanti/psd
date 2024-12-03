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

# Laporan Proyek 2
## Pengembangan Model Prediksi Kurs Jisdor Menggunakan Data Historis Time Series
## Pendahuluan

### Latar Belakang
Kurs JISDOR (Jakarta Interbank Spot Dollar Rate) merupakan acuan untuk nilai tukar resmi yang dikeluarkan Bank Indonesia berdasarkan transaksi antarbank di Indonesia. Kurs ini juga menjadi acuan penting di berbagai aktivitas ekonomi dan keuangan, termasuk perdagangan internasionall dan investasi.

### Tujuan Proyek
Proyek ini bertujuan untuk mengembangkan model prediksi kurs JISDOR berbasis data historis time series yang memiliki tingkat akurasi tinggi. Model prediksi ini diharapkan dapat:

•	Membantu pelaku usaha, investor, dan pembuat kebijakan dalam mengambil keputusan yang lebih baik terkait nilai tukar.

•	Memberikan wawasan strategis yang dapat digunakan untuk merancang kebijakan ekonomi yang lebih tanggap terhadap fluktuasi nilai tukar.

### Rumusan Masalah
•	Bagaimana cara mengembangkan model prediksi kurs JISDOR yang akurat dengan memanfaatkan data historis time series?

•	Bagaimana hasil prediksi kurs JISDOR dapat digunakan untuk mendukung pengambilan keputusan strategis dan memitigasi risiko terkait fluktuasi nilai tukar?

## METODOLOGI
### Data Understanding
#### a.	Sumber Data
Data yang digunakan dalam proyek ini diperoleh dari situs resmi Bank Indonesia, sebuah platform yang menyediakan data kurs JISDOR secara real-time dan historis. Di situs tersebut, tersedia informasi lengkap mengenai nilai tukar Rupiah terhadap Dolar Amerika Serikat berdasarkan transaksi antarbank.

Dalam proyek ini, digunakan data historis kurs JISDOR mulai dari tanggal 03 Agustus 2020 hingga 27 September 2024, yang telah diunduh dalam format CSV. Data ini mencakup nilai tukar harian yang menjadi dasar analisis untuk mengembangkan model prediksi menggunakan metode berbasis machine learning.


Untuk tampilan datanya bisa dilihat di bawah ini:
```{code-cell}
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data
df = pd.read_csv('https://raw.githubusercontent.com/DeviDuwiSusanti/dataset/main/kurs_jisdor.csv')

# mengubah kolom 'Date' dalam format datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'])

# Mengatur kolom 'Date' sebagai indeks
df.set_index('Tanggal', inplace=True)

# Mensortir data berdasarkan kolom tanggal dari terkecil ke terbesar
df = df.sort_values(by='Tanggal')
df
```

#### b.	Deskripsi Data Set
Data set ini terdiri dari 8 fitur atau kolom, dan 2230 record atau baris.
Atribut-atribut data set :
- Date		: tanggal data harga kurs berlaku, biasanya memiliki format YYYY-MM-DD.
- Kurs		: harga kurs pada tanggal tersebut.

```{code-cell}
df.info()
print('Ukuran data ', df.shape)
```
```{code-cell}
df.describe()
```

####  c. Eksplorasi Data
###### Mencari Missing Value
```{code-cell}
df.isnull().sum()
```
<!-- ###### Mencari Data yang Duplikat
```{code-cell}
duplicates = df[df.duplicated()]
print(duplicates)
``` -->
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
df['Low Target'] = df['Low'].shift(-1)

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
Dari dataset yang sudah melalui beberapa proses agar siap digunakan, terlihat bahwa fitur input terdiri dari fitur Open, High, Low, Close, Adj Close di hari ini, dan fitur Target Low atau prediksi harga Low besok hari sebagai fitur output.
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