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
Dataset terdiri dari 120 gambar yang diambil oleh kamera yang dipasang di kap mobil yang bergerak. Gambar-gambar ini mensimulasikan pemandangan jalan yang akan dihadapi oleh mobil otonom.

# Prediksi
Prediksi adalah video yang terdiri dari 120 gambar yang sama setelah diproses oleh model untuk menggambar kotak pembatas di sekitar objek yang terdeteksi.

# Cara Menggunakan
Letakkan gambar yang ingin Anda prediksi di folder images.
Jalankan sel berikut untuk memprediksi dan menyimpan hasilnya di folder out:
python
Copy code
out_scores, out_boxes, out_classes = predict(sess, "image_name.jpg")
Algoritma YOLO
YOLO ("You Only Look Once") adalah algoritma yang mendeteksi objek dalam satu kali forward pass melalui jaringan. Ini membuat YOLO sangat cepat dan efisien, hampir mencapai 45 frame per detik. Versi yang lebih kecil, Fast YOLO, dapat memproses hingga 155 frame per detik.
