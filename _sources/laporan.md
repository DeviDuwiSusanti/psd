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
Disini kita menjelaskan bagaimana menemukan data yang relevan untuk proyek kita, termasuk sumber data, cara memperolehnya, serta alat dan teknologi yang digunakan. Sumber data dapat berupa data internal Perusahaan, data dari situs web, atau data public yang tersedia secara online.

#### b.	Deskripsi Data Set
Deskripsi tentang dataset yang digunakan, seperti atribut-atribut dalam data, jenis data, serta gambaran umum tentang kualitas data.

### Data Preprocessing
Menjelaskan langkah-langkah yang dilakukan untuk pembersihan dan mempersiapkan data sebelum dimodelkan. Seperti menjelaskan tentang penanganan data yang hilang (missing values), tokenisasi dan normalisasi.

### Modelling
Menjelaskan proses pembuatan model berdasarkan data yang sudah kita proses
#### a.	Pembagian Data
Data dibagi menjadi dua, yaitu data pelatihan untuk melatih model dan data pengujian untuk mengecek seberapa baik model bekerja.

#### b.	Pembangunan Model
Pemilihan algoritma seperti SVM, Naïve Bayes, atau yang lainnya dan melatih model menggunakan data pelatihan untuk mengenali pola dalam data.

#### c.	Pengujian Model
Model diuji dengan data pengujian untuk melihat seberapa akurat prediksi yang dihasilkan.

### Evaluation
Di sini adalah tempat kita untuk mengukur kinerja model menggunakan metrik yang relevan seperi akurasi contohnya. Hal ini menentukan apakah model yang kita capai sudah memadai untuk digunakan dalam aplikasi nyata atau tidak.

## Kesimpulan

## Referensi