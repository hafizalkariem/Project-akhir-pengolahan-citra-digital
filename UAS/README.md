# K-Means Clustering dengan k=3 pada Gambar

Proyek ini menjelaskan cara melakukan segmentasi gambar menggunakan algoritma K-Means Clustering dengan jumlah cluster yang ditetapkan sebagai 3.

## Prasyarat

Pastikan Anda telah menginstal pustaka berikut:

- NumPy
- OpenCV
- Matplotlib
- Pillow (jika Anda perlu memeriksa metadata gambar)

Anda dapat menginstal pustaka ini menggunakan pip:

```bash
pip install numpy opencv-python matplotlib Pillow
```

## Langkah-langkah Implementasi

1. Membaca dan Menampilkan Gambar
   Baca gambar menggunakan OpenCV dan ubah warnanya dari BGR ke RGB untuk ditampilkan dengan benar menggunakan Matplotlib.

```py
import numpy as np
import matplotlib.pyplot as plt
import cv2
%matplotlib inline
# unutk membaca gambar gunakan gambar sesuai dengan yg dimiliki
image = cv2.imread('images/monarch.jpg')
# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
```

2. Membentuk Ulang Gambar dan Mengonversi Tipe Data
   Ubah bentuk gambar menjadi array 2D di mana setiap baris adalah piksel dan 3 nilai warna (RGB). Kemudian, konversi data piksel ke tipe float32.

```py
   #berfungsi untuk Membentuk ulang gambar menjadi susunan piksel 2D dan 3 nilaiwarna (RGB)
   pixel_vals = image.reshape((-1,3))
   # berfungsi untuk mengkonversikan ke tipe float
   pixel_vals = np.float32(pixel_vals)
```

3.  Menentukan Kriteria dan Menjalankan K-Means Clustering
    Tentukan kriteria untuk algoritma k-means dan atur jumlah cluster (k=3). Jalankan algoritma k-means.

```py
#baris kode di bawah ini menentukan kriteria agar algoritme berhenti berjalan,
#yang akan terjadi adalah 100 iterasi dijalankan atau epsilon (yang merupakanakurasi yang dibutuhkan)
#menjadi 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
# lalu lakukan k-means clustering dengan jumlah cluster yang ditetapkan sebagai 3
#juga pusat acak pada awalnya dipilih untuk pengelompokan k-means
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10,
cv2.KMEANS_RANDOM_CENTERS)
```

4. Memetakan Piksel ke Pusat Cluster dan Menampilkan Hasil
   Konversi pusat cluster ke tipe 8-bit, petakan setiap piksel ke pusat cluster yang sesuai, dan bentuk ulang data hasil clustering ke dimensi gambar asli. Tampilkan gambar hasil clustering.

python

```py

# mengonversi data menjadi nilai 8-bit
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
# membentuk ulang data menjadi dimensi gambar asli
segmented_image = segmented_data.reshape((image.shape))
plt.imshow(segmented_image)

```

## Menjelaskan Metadata Gambar

Untuk menjelaskan metadata gambar, kita bisa menggunakan library Pillow untuk mengekstrak dan menampilkan metadata EXIF.

Contoh Kode untuk Mengekstrak Metadata :

```py
from PIL import Image
from PIL.ExifTags import TAGS

# Membaca gambar
image_path = 'images/monarch.jpg'
image = Image.open(image_path)

# Mendapatkan metadata
exif_data = image._getexif()

# Menampilkan metadata
if exif_data:
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        print(f"{tag_name}: {value}")
else:
    print("Tidak ada metadata yang ditemukan")

```

## Penjelasan Metadata yang Umum

- **DateTime**: Tanggal dan waktu ketika gambar diambil.
- **Make dan Model**: Merek dan model kamera yang digunakan untuk mengambil gambar.
- **GPSInfo**: Informasi geografis yang menunjukkan lokasi di mana gambar diambil.
- **ExposureTime**: Waktu eksposur saat gambar diambil.
- **FNumber**: Nomor F, yang merupakan rasio panjang fokus lensa terhadap diameter pupil masuk.
- **ISOSpeedRatings**: Menunjukkan sensitivitas sensor kamera terhadap cahaya.

## Hasil

- [k-means clustering](k-means_clustering.ipynb)
- [metadata_definer](metadata_definer.ipynb)

## Kesimpulan

Proyek ini menunjukkan cara melakukan segmentasi gambar menggunakan K-Means Clustering dengan jumlah cluster yang ditetapkan sebagai 3. Selain itu, dijelaskan juga cara mengekstrak dan menampilkan metadata dari gambar.
