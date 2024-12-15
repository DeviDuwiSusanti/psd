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
## Pengembangan Model Prediksi Kurs Jual dan Beli Mata Uang JPY di Bank Indonesia Menggunakan Data Historis
## Pendahuluan

### Latar Belakang
Kurs mata uang adalah salah satu indikator ekonomi yang memiliki peran penting dalam aktivitas perdagangan internasional, investasi, dan pengelolaan keuangan negara. Di Indonesia, Bank Indonesia (BI) menerbitkan kurs referensi harian untuk berbagai mata uang, termasuk kurs jual dan kurs beli mata uang asing seperti Yen Jepang (JPY).

Kurs jual adalah nilai yang digunakan oleh bank untuk menjual mata uang asing kepada konsumen, sedangkan kurs beli adalah nilai yang digunakan bank untuk membeli mata uang asing dari konsumen. Fluktuasi kurs jual dan kurs beli dipengaruhi oleh berbagai faktor, seperti kondisi ekonomi global, kebijakan moneter, serta tingkat permintaan dan penawaran di pasar valuta asing.

Bagi pelaku usaha, investor, dan pemangku kepentingan lainnya, memprediksi pergerakan kurs jual dan kurs beli mata uang asing dapat memberikan keuntungan strategis. Oleh karena itu, diperlukan teknologi prediktif seperti machine learning yang memanfaatkan data historis untuk memberikan estimasi kurs di masa mendatang.

### Tujuan Proyek
Proyek ini bertujuan untuk mengembangkan model prediksi kurs jual dan kurs beli mata uang JPY berdasarkan data historis yang diterbitkan oleh Bank Indonesia. Model ini diharapkan dapat:
- Membantu pelaku usaha dan investor dalam membuat keputusan keuangan yang lebih baik.
- Memberikan wawasan mengenai pola fluktuasi kurs JPY.
- Meminimalkan risiko yang terkait dengan perubahan nilai tukar.

### Rumusan Masalah
- Bagaimana membangun model prediksi kurs jual dan kurs beli JPY yang akurat menggunakan data historis?
- Bagaimana hasil prediksi kurs dapat mendukung pengambilan keputusan strategis di sektor keuangan dan bisnis?

## METODOLOGI
### Data Understanding
#### a.	Sumber Data
Data yang digunakan pada proyek ini diperoleh dari website https://datacenter.ortax.org/ortax/kursbi/show/JPY, yaitu sebuah platform online yang menyediakan data keuangan dan pasar valuta asing secara real-time. Di website tersebut, tersedia data historis kurs jual dan kurs beli mata uang JPY yang diterbitkan oleh Bank Indonesia. Data ini mencakup periode dari tanggal 01-01-2020 hingga 06-12-2024 dan diunduh dalam format dokumen CSV untuk keperluan analisis lebih lanjut.

Untuk tampilan datanya bisa dilihat di bawah ini:
```{code-cell}
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data
df = pd.read_csv('https://raw.githubusercontent.com/DeviDuwiSusanti/dataset/refs/heads/main/JPY_updated.csv')

# mengubah kolom 'Tanggal' dalam format datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'])

# Mengatur kolom 'Tanggal' sebagai indeks
df.set_index('Tanggal', inplace=True)

# Mensortir data berdasarkan kolom Tanggal dari terkecil ke terbesar
df = df.sort_values(by='Tanggal')
df
```

#### b.	Deskripsi Data Set
Data set ini terdiri dari 6 fitur atau kolom, dan 1802 record atau baris.
Atribut-atribut data set :
- Date: Tanggal data kurs JPY, biasanya dalam format YYYY-MM-DD.
- Kurs Jual: Nilai tukar yang digunakan untuk menjual JPY.
- Kurs Beli: Nilai tukar yang digunakan untuk membeli JPY.

```{code-cell}
df.info()
print('Ukuran data ', df.shape)
```
```{code-cell}
df[['Kurs Jual', 'Kurs Beli']].describe()
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
df.plot()
```

###### Deteksi Outlier
```{code-cell}
sns.boxplot(data=df)
```
###### Korelasi Antar Fitur
```{code-cell}
correlation_matrix = df.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```
Dari heatmap di atas, bisa dilihat bahwa:
Semua fitur memiliki hubungan yang kuat satu sama lain.

### Data Preprocessing
Langkah-langkah pada tahap ini adalah sebagai berikut :

#### a. Normalisasi
```{code-cell}
import numpy as np

def normalize(df):
    from sklearn.preprocessing import RobustScaler, MinMaxScaler

    np_data_unscaled = np.array(df)
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)
    print(np_data_unscaled)
    normalized_df = pd.DataFrame(np_data_scaled, columns=df.columns, index=df.index)
    pd.set_option('display.float_format', '{:.4f}'.format)  # Menampilkan 4 desimal
    return normalized_df, scaler

normalized_df, scaler = normalize(df)
normalized_df
```

#### b. Sliding Windows
Untuk memprediksi transaksi Bank Indonesia terhadap mata uang JPY beberapa hari ke depan dibutuhkan data beberapa hari yang lalu.
```{code-cell}
import pandas as pd

def sliding_window(data, lag):
    series_sell = data['Kurs Jual']
    series_buy = data['Kurs Beli']
    result = pd.DataFrame()

    # Menambahkan kolom lag untuk 'sell'
    for l in lag:
        result[f'Kurs Jual-{l}'] = series_sell.shift(l)
    
    # Menambahkan kolom lag untuk 'buy'
    for l in lag:
        result[f'Kurs Beli-{l}'] = series_buy.shift(l)
    result

    # Menambahkan kolom 'sell' dan 'buy' asli tanpa perubahan
    result['Kurs Jual'] = series_sell[l:]
    result['Kurs Beli'] = series_buy[l:]
    
    # Menghapus nilai yang hilang (NaN)
    result = result.dropna()
    
    # Mengatur index sesuai dengan nilai lag
    result.index = series_sell.index[l:]
    
    return result

windowed_data = sliding_window(normalized_df, [1, 2, 3, 4, 5, 6, 7])
windowed_data = windowed_data[['Kurs Jual', 'Kurs Jual-1', 'Kurs Jual-2', 'Kurs Jual-3', 'Kurs Jual-4', 'Kurs Jual-5', 'Kurs Jual-6', 'Kurs Jual-7', 'Kurs Beli', 'Kurs Beli-1', 'Kurs Beli-2', 'Kurs Beli-3', 'Kurs Beli-4', 'Kurs Beli-5', 'Kurs Beli-6', 'Kurs Beli-7']]
print(windowed_data)
```

#### c. Splitting Data
```{code-cell}
def split_data(data, target, train_size):
    split_index = int(len(data) * train_size)

    x_train = data[:split_index]
    y_train = target[:split_index]
    x_test = data[split_index:]
    y_test = target[split_index:]

    return x_train, y_train, x_test, y_test

input_df = windowed_data.drop(columns=['Kurs Jual', 'Kurs Beli'])
target_df = windowed_data[['Kurs Jual', 'Kurs Beli']]

x_train, y_train, x_test, y_test = split_data(input_df, target_df, 0.8)

print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
```

### Modelling
#### Membangun Model
##### Regresi Linear
```{code-cell}
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
```

```{code-cell}
# Melakukan prediksi
y_pred = linear_model.predict(x_test)

# Menghitung error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape, " %")
```

```{code-cell}
import matplotlib.pyplot as plt

# Membuat grafik perbandingan
plt.figure(figsize=(10, 6))

# Nilai aktual
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_test.values)[:, 0], 
    color='blue', marker='o', linestyle='-', markersize=4, label='Aktual Sell'
)
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_test.values)[:, 1], 
    color='green', marker='o', linestyle='-', markersize=4, label='Aktual Buy'
)

# Nilai prediksi
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_pred)[:, 0], 
    color='orange', marker='x', linestyle='--', markersize=4, label='Prediksi Sell'
)
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_pred)[:, 1], 
    color='red', marker='x', linestyle='--', markersize=4, label='Prediksi Buy'
)

# Menambahkan judul dan label sumbu
plt.title('Perbandingan Nilai Aktual dan Prediksi Model Regresi Linear')
plt.xlabel('Tanggal')
plt.ylabel('Harga')

# Menampilkan grid
plt.grid()

# Menambahkan legenda
plt.legend()

# Menampilkan grafik
plt.show()
```

##### Regresi Linear + Ensamble Bagging
```{code-cell}
from sklearn.ensemble import BaggingRegressor
base_model = LinearRegression()
bagging_model = BaggingRegressor(estimator=base_model, n_estimators=10, bootstrap=True)
bagging_model.fit(x_train, y_train)
y_pred = bagging_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Percentage Error (MAPE): {mape} %')
```

```{code-cell}
import matplotlib.pyplot as plt

# Membuat grafik perbandingan
plt.figure(figsize=(10, 6))

# Nilai aktual
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_test.values)[:, 0], 
    color='blue', marker='o', linestyle='-', markersize=4, label='Aktual Sell'
)
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_test.values)[:, 1], 
    color='green', marker='o', linestyle='-', markersize=4, label='Aktual Buy'
)

# Nilai prediksi
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_pred)[:, 0], 
    color='orange', marker='x', linestyle='--', markersize=4, label='Prediksi Sell'
)
plt.plot(
    y_test.index, 
    scaler.inverse_transform(y_pred)[:, 1], 
    color='red', marker='x', linestyle='--', markersize=4, label='Prediksi Buy'
)

# Menambahkan judul dan label sumbu
plt.title('Perbandingan Nilai Aktual dan Prediksi Model Regresi Linear')
plt.xlabel('Tanggal')
plt.ylabel('Harga')

# Menampilkan grid
plt.grid()

# Menambahkan legenda
plt.legend()

# Menampilkan grafik
plt.show()
```

### Evaluation
Dari model-model di atas didapatkan bahwa model Regresi Linear lebih baik dibandingkan Regresi Linear ditambah degan Ensamble Bagging.

### Deployment
Hasil deployment dapat diakses di link berikut ini : https://huggingface.co/spaces/heviaa/projek3_prediksiXRP
