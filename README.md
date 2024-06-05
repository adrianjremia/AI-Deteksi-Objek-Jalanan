# ProjectPratikumAI

Identifikasi Objek jalanan 

Pendahuluan
Proyek ini bertujuan untuk mendeteksi objek di jalan menggunakan algoritma YOLO (You Only Look Once) untuk membantu dalam mengemudi otonom. YOLO adalah algoritma deteksi objek real-time yang populer karena kemampuannya mencapai akurasi tinggi dengan kecepatan yang sangat cepat.

Dataset
Dataset terdiri dari 120 gambar yang diambil oleh kamera yang dipasang di kap mobil yang bergerak. Gambar-gambar ini mensimulasikan pemandangan jalan yang akan dihadapi oleh mobil otonom.

Prediksi
Prediksi adalah video yang terdiri dari 120 gambar yang sama setelah diproses oleh model untuk menggambar kotak pembatas di sekitar objek yang terdeteksi.

Cara Menggunakan
Letakkan gambar yang ingin Anda prediksi di folder images.
Jalankan sel berikut untuk memprediksi dan menyimpan hasilnya di folder out:
python
Copy code
out_scores, out_boxes, out_classes = predict(sess, "image_name.jpg")
Algoritma YOLO
YOLO ("You Only Look Once") adalah algoritma yang mendeteksi objek dalam satu kali forward pass melalui jaringan. Ini membuat YOLO sangat cepat dan efisien, hampir mencapai 45 frame per detik. Versi yang lebih kecil, Fast YOLO, dapat memproses hingga 155 frame per detik.

Detail Model
Input: Batch gambar dengan bentuk (m, 608, 608, 3)
Output: Daftar kotak pembatas bersama dengan kelas yang dikenali. Setiap kotak pembatas diwakili oleh 6 angka (p_c, b_x, b_y, b_h, b_w, c). Jika c dikembangkan menjadi vektor 80 dimensi, setiap kotak pembatas diwakili oleh 85 angka.
Arsitektur YOLO: Jika menggunakan 5 anchor boxes: GAMBAR (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85). Model memprediksi total 19x19x5 = 1805 kotak dalam satu kali forward pass.
Penyaringan
Untuk mengurangi jumlah objek yang terdeteksi, dua teknik diterapkan:

Score-thresholding: Buang kotak yang mendeteksi kelas dengan skor di bawah ambang batas.
Non-maximum suppression (NMS): Pilih hanya kotak dengan skor tertinggi dari beberapa prediksi untuk objek yang sama.

Langkah-langkah NMS:
Pilih kotak dengan skor tertinggi.
Hitung tumpang tindih dengan semua kotak lainnya dan hapus kotak yang tumpang tindih lebih dari iou_threshold.
Ulangi sampai tidak ada lagi kotak dengan skor lebih rendah dari kotak yang dipilih saat ini.

Gambar Input:
Gambar Output:
