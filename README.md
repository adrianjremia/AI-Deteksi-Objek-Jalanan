# Project Pratikum AI - Identifikasi Objek jalanan 

# Author
- Muhammad Wildan Kamil 220009
- Reymontha Tarigan 220021
- Adrian Jeremia Kurniawan 220047
- Muhammad Exsfo al Banjari 220057
- Tegar Simanjuntak 220085

# Pendahuluan
Proyek ini bertujuan untuk mendeteksi objek di jalan menggunakan algoritma YOLO (You Only Look Once) untuk membantu dalam mengemudi otonom. YOLO adalah algoritma deteksi objek real-time yang populer karena kemampuannya mencapai akurasi tinggi dengan kecepatan yang sangat cepat.

# Kegunaan
Program kami ini akan mendeteksi berbagai kendaraan yang ada di jalanan seperti mobil, bus, truk, sepeda, dan motor. 

# Dataset
Dataset terdiri dari 1200 gambar yang diambil oleh kamera yang dipasang di kap mobil yang bergerak. Gambar-gambar ini mensimulasikan pemandangan jalan yang akan dihadapi oleh mobil otonom.

# Prediksi
Prediksi adalah video yang terdiri dari 1200 gambar yang sama setelah diproses oleh model untuk menggambar kotak pembatas di sekitar objek yang terdeteksi.

# Cara Kerja
Upload image yang ingin dideteksi.

Kemudian program akan menlatih dengan:
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5m.pt

Perintah di atas akan melatih model dengan parameter berikut:
- --img 640: Ukuran gambar 640x640 piksel.
- --batch 16: Ukuran batch 16.
- --epochs 50: Jumlah epoch 50.
- --data dataset.yaml: File konfigurasi dataset.
- --weights yolov5m.pt: Menggunakan bobot model YOLOv5m yang sudah di-pretrained.

Setelah pelatihan selesai, bobot hasil pelatihan akan disimpan dan dapat digunakan untuk pengujian.

python detect.py --weights runs/train/exp12/weights/best.pt --source test_images/imtest13.JPG


Perintah di atas akan menggunakan bobot hasil pelatihan dari path runs/train/exp12/weights/best.pt untuk mendeteksi objek pada gambar test_images/imtest13.JPG.

# Hasil Running

![gambar](https://github.com/adrianjremia/AI-Deteksi-Objek-Jalanan/blob/main/imtest.JPG)
![gambar](https://github.com/adrianjremia/AI-Deteksi-Objek-Jalanan/blob/main/imtest13.JPG)


