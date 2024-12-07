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
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
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



####  c. Eksplorasi Data
###### Mencari Missing Value
```{code-cell}
df.isnull().sum()
```
Selanjutnya, dikarenakan dihari libur seperti sabtu dan minggu, data tidak tersedia, maka perlu diberlakukannya interpolasi, berikut ini adalah codenya:
```{code-cell}
# Melengkapi tanggal yang terlewat dengan frekuensi harian
df = df.asfreq('D')

# Mengisi nilai yang hilang menggunakan interpolasi linear
df.interpolate(method='linear', inplace=True)

# metode linear :
# Menampilkan data setelah interpolasi
df
df.plot()
```
```{code-cell}
df.info()
print('Ukuran data ', df.shape)
```
```{code-cell}
df.describe()
```

### Data Preprocessing
#### Sliding Window
```{code-cell}
def s_windows(jumlah):
    for i in range(1, jumlah):
        df[f'Kurs-{i}'] = df['Kurs'].shift(i)
    df.dropna(inplace=True)
    return df

df = s_windows(4)
df
```

### Normalisasi Data
```{code-cell}
# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Kurs-1, Kurs-2, Kurs-3, Kurs-4, Kurs-5)
df_features_normalized = pd.DataFrame(scaler_features.fit_transform(df[['Kurs-1', 'Kurs-2', 'Kurs-3']]),
                                      columns=['Kurs-1', 'Kurs-2', 'Kurs-3'],
                                      index=df.index)

# Normalisasi target (Kurs)
df_target_normalized = pd.DataFrame(scaler_target.fit_transform(df[['Kurs']]),
                                    columns=['Kurs'],
                                    index=df.index)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_target_normalized, df_features_normalized], axis=1)
df_normalized.head()
```

### Modelling
Menjelaskan proses pembuatan model berdasarkan data yang sudah kita proses
##### a.	Pembagian Data
Data dibagi menjadi dua, yaitu data pelatihan untuk melatih model dan data pengujian untuk mengecek seberapa baik model bekerja.
```{code-cell}
# Mengatur fitur (X) dan target (y)
X = df_normalized[['Kurs-1', 'Kurs-2', 'Kurs-3']]
y = df_normalized['Kurs']

# Membagi data menjadi training dan testing (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

##### b.	Pembangunan Model
Di sini dilakukan percobaan dnegan 3 model ditambah dengan ensamble bagging, yaitu Regresi Linear, SVR, dan KNN
```{code-cell}
# List model untuk ensemble Bagging
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Dictionary untuk menyimpan hasil evaluasi
results = {}

# Iterasi setiap model
for i, (name, base_model) in enumerate(models.items()):
    # Inisialisasi Bagging Regressor
    bagging_model = BaggingRegressor(
        estimator=base_model, 
        n_estimators=10, 
        max_samples=0.7, 
        random_state=32, 
        bootstrap=True
    )
    
    # Latih model
    bagging_model.fit(X_train, y_train)
    
    # Prediksi pada data uji
    y_pred = bagging_model.predict(X_test)
    
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

## Kesimpulan

## Referensi