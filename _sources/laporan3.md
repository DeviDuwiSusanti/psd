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

# Laporan Proyek 3
## Pengembangan Model Prediksi Harga Cryptocurrency Etherium (ETH) Menggunakan Data Historis untuk Mendukung Keputusan Investasi
## Pendahuluan

### Latar Belakang
Cryptocurrency adalah aset digital yang menggunakan teknologi blockchain untuk mencatat transaksi secara transparan dan aman. Salah satu cryptocurrency yang berkembang pesat adalah Solana (SOL), sebuah platform blockchain yang dikenal karena kecepatan transaksi dan biaya rendah. Solana telah menarik perhatian investor karena potensinya dalam mendukung aplikasi terdesentralisasi (dApps) dan proyek berbasis blockchain lainnya.

Namun, seperti aset cryptocurrency lainnya, harga Solana sangat fluktuatif, dipengaruhi oleh berbagai faktor seperti sentimen pasar, perkembangan teknologi, regulasi, dan kondisi ekonomi global. Fluktuasi ini sering kali menyulitkan investor untuk membuat keputusan investasi yang tepat.

Untuk membantu investor memahami pergerakan harga Solana, diperlukan teknologi yang dapat memprediksi harga di masa depan, seperti machine learning. Dengan analisis berbasis data historis, teknologi ini dapat membantu meminimalkan risiko dan mendukung pengambilan keputusan investasi yang lebih baik.

### Tujuan Proyek
Proyek ini bertujuan untuk mengembangkan model prediksi harga cryptocurrency Solana (SOL) menggunakan data historis. Analisis ini diharapkan dapat:

- Membantu investor dalam membuat keputusan investasi yang lebih terinformasi.
- Memberikan wawasan terkait potenonnsi pergerakan harga Solana untuk memaksimalkan keuntungan dan mengelola risiko dengan lebih baik.

### Rumusan Masalah
- Bagaimana mengembangkan model prediksi harga Solana (SOL) yang akurat dengan memanfaatkan data historis?
- Bagaimana hasil prediksi harga Solana dapat digunakan untuk mendukung keputusan investasi yang lebih baik di pasar cryptocurrency?

## METODOLOGI
### Data Understanding
#### a.	Sumber Data
Data yang digunakan pada proyek ini diperoleh dari website https://finance.yahoo.com/quote/SOL-USD/history/, yaitu sebuah platform online yang menyediakan data keuangan dan pasar aset secara real-time. Di website tersebut, tersedia informasi atau data historis harga cryptocurrency Solana (SOL) dalam berbagai rentang waktu.

Pada proyek ini, data historis yang digunakan mencakup periode dari tanggal 10-04-2020 hingga 05-12-2024, yang diunduh dalam format dokumen CSV.

Untuk tampilan datanya bisa dilihat di bawah ini:
```{code-cell}
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data
df = pd.read_csv('https://raw.githubusercontent.com/DeviDuwiSusanti/dataset/refs/heads/main/etherium.csv')

# mengubah kolom 'Date' dalam format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Mengatur kolom 'Date' sebagai indeks
df.set_index('Date', inplace=True)

# Mensortir data berdasarkan kolom Date dari terkecil ke terbesar
df = df.sort_values(by='Date')
df
```

#### b.	Deskripsi Data Set
Data set ini terdiri dari 8 fitur atau kolom, dan 2230 record atau baris.
Atribut-atribut data set :
- Date		: tanggal data harga aset koin, biasanya memiliki format YYYY-MM-DD.
- Open		: harga pembukaan aset koin pada tanggal tersebut.
- High		: harga tertinggi yang dicapai pada tanggal tersebut.
- Low		: harga terendah aset koin pada tanggal tersebut.
- Close		: harga penutupan aset koin pada tanggal tersebut.
- Adj Close	: harga penutupan yang sudah disesuaikan dengan pembagian aset koin,     
  		       dividen, dan coerporate actions lainnya.
- Volume	: jumlah aset koin yang diperdagangkan pada tanggal tersebut.

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

###### Menampilkan Trend Setiap Fitur
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

###### Korelasi Antar Fitur
```{code-cell}
correlation_matrix = df.corr()

plt.figure(figsize=(7, 3))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()
```
Dari heatmap di atas, bisa dilihat bahwa:
Fitur pembukaan (Open), tertinggi (High), penutupan (Close), dan harga pennutupan yang disesuaikan (Adj Close) mempunyai korelasi yang kuat antara satu sama lain (mendekati 1 atau 1). Hal ini menunjukkan fitur-fitur tersebut saling berkaitan dan bergerak sejalan. Sedangkan fitur 'Volume' mempunyai korelasi paling rendah dengan fitur lainnya (sekitar 0.78 - 0.8) yang menunjukkan bahwa perubahan volume tidak berpengaruh langsung dengan perubahan harga. Sehingga fitur volume tidak perlu digunakan untuk analisis prediksi pada projek ini.

### Data Preprocessing
Langkah-langkah pada tahap ini adalah sebagai berikut :

#### a. Menghapus Fitur yang tidak relevan
Pada tahap proses menghitung matriks korelasi, didapat bahwa fitur 'volume' tidak relevan atau tidak memiliki pengarus terhadap fitur lainnya, maka fitur 'volume' akan dihapus. Serta fitur 'Adj Close' dimana nilai dari fitur ini sama dengan fitur 'Close'.
```{code-cell}
df = df.drop(columns=['Volume', 'Adj Close'])
df.head()
```

#### b. Rekayasa Fitur
Karena dalam penelitian ini kita akan memprediksi harga Close pada hari berikutnya, maka perlu variabel baru untuk target. Dimana fitur ini dapat membantu kita untuk mengetahui seberapa rendah harga saham bisa turun. Para investor juga bisa menggunakan prediksi ini untuk membeli aset saat harganya rendah, dan meningkatkan peluang mendapatkan keuntungan saat harga saham naik lagi.
```{code-cell}
df['Close Target'] = df['Close'].shift(-1)

df = df[:-1]
df.head()
```
Dari dataset yang sudah melalui beberapa proses agar siap digunakan, terlihat bahwa fitur input terdiri dari fitur Open, High, Low, Close, Adj Close di hari ini, dan fitur Close Target atau prediksi harga Low besok hari sebagai fitur output.


#### c. Normalisasi Data
```{code-cell}

# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Open, High, Low,, 'Close' Close Target-4, Close Target-5)
df_features_normalized = pd.DataFrame(scaler_features.fit_transform(df[['Open', 'High', 'Low', 'Close']]),
                                      columns=['Open', 'High', 'Low', 'Close'],
                                      index=df.index)

# Normalisasi target (Close Target)
df_target_normalized = pd.DataFrame(scaler_target.fit_transform(df[['Close Target']]),
                                    columns=['Close Target'],
                                    index=df.index)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_features_normalized, df_target_normalized], axis=1)
df_normalized.head()
```

### Modelling

#### Pembagian Data
```{code-cell}
# Mengatur fitur (X) dan target (y)
X = df_normalized[['Open', 'High', 'Low', 'Close']]
y = df_normalized['Close Target']

# Membagi data menjadi training dan testing (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
```

#### Membangun Model
```{code-cell}
# List model regresi
models = {
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=32),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Dictionary untuk menyimpan hasil evaluasi
results = {}

# Iterasi setiap model
for name, model in models.items():
    # Latih model
    model.fit(X_train, y_train)
    
    # Prediksi pada data uji
    y_pred = model.predict(X_test)
    
    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Dalam persen
    
    # Simpan hasil evaluasi
    results[name] = {"RMSE": rmse, "MAPE": mape}
    
    # Kembalikan hasil prediksi ke skala asli
    y_pred_original = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original = scaler_target.inverse_transform(y_test.values.reshape(-1, 1))
    
    # Plot hasil prediksi
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test_original, label="Actual", color="blue")
    plt.plot(y_test.index, y_pred_original, label=f"Predicted ({name})", color="red")
    
    # Tambahkan detail plot
    plt.title(f'Actual vs Predicted Values ({name})')
    plt.xlabel('Tanggal')
    plt.ylabel('Kurs')
    plt.legend()
    plt.grid(True)
    
    # Tampilkan plot
    plt.show()

# Tampilkan hasil evaluasi
print("HASIL EVALUASI MODEL")
for model, metrics in results.items():
    print(f"{model}:\n  RMSE: {metrics['RMSE']:.2f}\n  MAPE: {metrics['MAPE']:.2f}%\n")

```

### Evaluation
Di sini adalah tempat kita untuk mengukur kinerja model menggunakan metrik yang relevan seperi akurasi contohnya. Hal ini menentukan apakah model yang kita capai sudah memadai untuk digunakan dalam aplikasi nyata atau tidak.

## Kesimpulan

## Referensi