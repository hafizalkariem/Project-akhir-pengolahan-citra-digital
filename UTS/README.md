# Face eye detection

Selamat datang di Aplikasi Deteksi Wajah dan Mata! Aplikasi ini memungkinkan Anda untuk mengunggah gambar dan secara otomatis mendeteksi wajah dan mata dalam gambar tersebut menggunakan Haar cascades. Dibangun dengan OpenCV dan Streamlit, aplikasi ini menyediakan antarmuka yang sederhana namun kuat untuk pemrosesan gambar secara real-time.

## Fitur

Deteksi Wajah: Mengidentifikasi dan menandai wajah dalam gambar.
Deteksi Mata: Mendeteksi dan menandai mata dalam wajah yang terdeteksi.
Antarmuka Pengguna yang Ramah: Unggah gambar dengan mudah dan lihat hasilnya langsung di browser Anda.
Memulai
Ikuti langkah-langkah berikut untuk mendapatkan salinan proyek ini dan menjalankannya di mesin lokal Anda.

## Prasyarat

Pastikan Anda memiliki Python terinstal di sistem Anda. Disarankan untuk menggunakan virtual environment untuk mengelola dependensi proyek.

## App.py

```
import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Load Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def detect_faces_and_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return img, len(faces)

st.title("Face and Eye Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    detected_img, face_count = detect_faces_and_eyes(img_array)

    if face_count == 0:
        st.write("No faces detected.")
    else:
        st.write(f"Detected {face_count} face(s).")

    st.image(detected_img, caption='Processed Image', use_column_width=True)
else:
    st.text("Upload an image to get started.")
```

## Hasil projek

<img width="956" alt="image" src="https://github.com/afrizalfajrianto/UTS_Pengolahan_Citra/assets/115614098/c55c90d4-bc88-4af3-8a1e-96015877bae5">
