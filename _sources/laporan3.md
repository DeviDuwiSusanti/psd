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
Terlihat bahwa fitur pada data ini tidak memiliki outlier.

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

#### a. Sliding Window
ntuk memprediksi transaksi Bank Indonesia terhadap mata uang JPY beberapa hari ke depan dibutuhkan data beberapa hari yang lalu.
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

windowed_data = sliding_window(df, [1, 2, 3])
windowed_data = windowed_data[['Kurs Jual', 'Kurs Jual-1', 'Kurs Jual-2', 'Kurs Jual-3', 'Kurs Beli', 'Kurs Beli-1', 'Kurs Beli-2', 'Kurs Beli-3']]
windowed_data
```

#### b. Normalisasi
```{code-cell}
from sklearn.preprocessing import MinMaxScaler

# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Kurs Jual-1, Kurs Jual-2, Kurs Jual-3, Kurs Beli-1, Kurs Beli-2, Kurs Beli-3)
df_features_normalized = pd.DataFrame(
    scaler_features.fit_transform(windowed_data[['Kurs Jual-1', 'Kurs Jual-2', 'Kurs Jual-3', 'Kurs Beli-1', 'Kurs Beli-2', 'Kurs Beli-3']]),
    columns=['Kurs Jual-1', 'Kurs Jual-2', 'Kurs Jual-3', 'Kurs Beli-1', 'Kurs Beli-2', 'Kurs Beli-3'],
    index=windowed_data.index
)

# Normalisasi target (Kurs Jual dan Kurs Beli)
df_target_normalized = pd.DataFrame(
    scaler_target.fit_transform(windowed_data[['Kurs Jual', 'Kurs Beli']]),
    columns=['Kurs Jual', 'Kurs Beli'],
    index=windowed_data.index
)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_features_normalized, df_target_normalized], axis=1)
df_normalized.head()
```

#### c. Splitting Data
```{code-cell}
# Mengatur fitur (X) dan target (y)
X = df_normalized[['Kurs Jual-1', 'Kurs Jual-2', 'Kurs Jual-3', 'Kurs Beli-1', 'Kurs Beli-2', 'Kurs Beli-3']]
y = df_normalized[['Kurs Jual', 'Kurs Beli']]

# Membagi data menjadi training dan testing (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
```

### Modelling
#### Membangun Model
```{code-cell}
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=32),
    "Ridge Regression": Ridge(alpha=1.0)
}

# Dictionary untuk menyimpan hasil evaluasi
results = {}

# Iterasi setiap model
for name, model in models.items():
    # Latih model
    model.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = model.predict(X_test)

    # Evaluasi untuk setiap target (Kurs Jual dan Kurs Beli)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Dalam persen

    # Simpan hasil evaluasi
    results[name] = {"RMSE": rmse, "MAPE": mape}

    # Kembalikan hasil prediksi ke skala asli untuk kedua target
    y_pred_original = scaler_target.inverse_transform(y_pred)
    y_test_original = scaler_target.inverse_transform(y_test)

    # Plot hasil prediksi untuk Kurs Jual dan Kurs Beli dalam satu plot
    plt.figure(figsize=(15, 6))

    # Plot untuk Kurs Jual dan Kurs Beli
    plt.plot(y_test.index, y_test_original[:, 0], label="Actual Kurs Jual", color="blue", linestyle='-')
    plt.plot(y_test.index, y_pred_original[:, 0], label=f"Predicted Kurs Jual ({name})", color="red", linestyle='--')
    
    plt.plot(y_test.index, y_test_original[:, 1], label="Actual Kurs Beli", color="green", linestyle='-')
    plt.plot(y_test.index, y_pred_original[:, 1], label=f"Predicted Kurs Beli ({name})", color="orange", linestyle='--')

    # Tambahkan detail plot
    plt.title(f'Actual vs Predicted Kurs Jual and Kurs Beli ({name})')
    plt.xlabel('Tanggal')
    plt.ylabel('Kurs')
    plt.legend()
    plt.grid(True)

    # Tampilkan plot
    plt.show()

# Tampilkan hasil evaluasi
print("HASIL EVALUASI MODEL")
for model, metrics in results.items():
    print(f"{model}:\n  RMSE: {metrics['RMSE']}\n ")

# Menentukan model terbaik berdasarkan RMSE atau MAPE (misalnya RMSE terendah)
best_model_name = min(results, key=lambda x: results[x]['RMSE'])  # Model dengan RMSE terendah
best_model_rmse = results[best_model_name]['RMSE']

```
### Evaluation
Dari model-model di atas didapatkan bahwa model Regresi Linear lebih baik dibandingkan dengan model Decision Tree dan Ridge Regression.

### Deployment
Hasil deployment dapat diakses di link berikut ini : https://huggingface.co/spaces/heviaa/projek3_prediksiXRP
