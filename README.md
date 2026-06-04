  FRUIT NINJA CV USING COMPUTER VISION
=================================================

Version     : 1.0.0
Language    : Python 3.11
Category    : Computer Vision / Game
Framework   : OpenCV, MediaPipe, Pygame

  PROJECT OVERVIEW
=================================================

Project ini merupakan implementasi game Fruit Ninja
berbasis Computer Vision sebagai Tugas Akhir Semester
(UAS) mata kuliah Data Sains.

Pemain menggunakan jari telunjuk sebagai "pedang"
virtual yang ditangkap kamera secara real-time untuk
menebas buah-buahan yang melayang di layar.

Teknologi utama yang digunakan:

1. MediaPipe Hand Landmarker
   - Mendeteksi 21 titik landmark tangan secara real-time.
   - Melacak posisi ujung jari telunjuk (landmark ke-8).
   - Berjalan dalam mode LIVE_STREAM untuk latensi rendah.

2. Finger Detection Algorithm
   - Mendeteksi apakah telunjuk sedang terangkat.
   - Membandingkan ekstensi tiap jari terhadap MCP-nya.
   - Menggunakan margin threshold untuk mencegah false positive.

3. FingerTracker dengan Velocity Prediction
   - Menyimpan riwayat posisi jari menggunakan deque.
   - Menghitung kecepatan dan arah gerakan jari secara real-time.
   - Memprediksi posisi masa depan untuk deteksi slice lebih akurat.
   - Smoothing adaptif berdasarkan kecepatan gerakan jari.

4. Slice Detection
   - Mengecek collision antara trail jari dan objek buah/bom.
   - Menggunakan segment-to-point distance untuk akurasi tinggi.
   - Memanfaatkan predicted position untuk menangkap gerakan cepat.

=================================================
  GAME OBJECTIVE
=================================================

Tujuan pemain adalah:

- Menebas buah yang melayang menggunakan jari telunjuk.
- Menghindari menebas BOM (langsung game over).
- Memaksimalkan score dengan combo beruntun.
- Menjaga nyawa agar tidak habis (3 nyawa awal).
- Memecahkan high score pribadi.

=================================================
  PROJECT ARCHITECTURE
=================================================

+---------------------------+
|         game.py           |
+------------+--------------+
             |
     +-------+-------+
     |               |
     v               v
MediaPipe        Pygame Mixer
Hand Landmarker  (Audio Engine)
     |
     v
+---------------------------+
|     is_index_raised()     |  <-- Deteksi telunjuk
+---------------------------+
     |
     v
+---------------------------+
|      FingerTracker        |  <-- Smoothing + prediksi
+---------------------------+
     |
     v
+---------------------------+
|      check_slice()        |  <-- Deteksi tebasan
+---------------------------+
     |
     v
+---------------------------+
|      GameObject           |  <-- Buah / Bom
+---------------------------+
     |
     v
  Game Loop (OpenCV Window)

=================================================
  DIRECTORY STRUCTURE
=================================================

Fruit_Ninja_UAS/
│
├── assets/
│   ├── apple.png
│   ├── orange.png
│   ├── watermelon.png
│   ├── kiwi.png
│   ├── strawberry.png
│   ├── peach.png
│   ├── bomb.png
│   ├── Clean-Slice-1.mp3
│   ├── Game-over.mp3
│   ├── Game-start.mp3
│   ├── Throw-bomb.mp3
│   └── Throw-fruit.mp3
│
├── game.py                  <-- File utama game
├── hand_landmarker.task     <-- Model MediaPipe
├── highscore.txt            <-- Dibuat otomatis
├── build_exe.bat            <-- Build menjadi .exe
├── setup_and_run.bat        <-- Auto setup Windows
├── setup_and_run.sh         <-- Auto setup Mac/Linux
└── README.md

=================================================
  QUICK START — 2 CARA MENJALANKAN
=================================================

----------------------------------------------
  CARA 1: LANGSUNG MAIN (Tanpa Install Apapun)
----------------------------------------------

  Khusus Windows — semua sudah ter-bundle:

  1. Download file ZIP di bagian Releases GitHub
     (klik tab "Releases" di kanan halaman repo)
  2. Extract ZIP ke folder mana saja
  3. Buka folder hasil extract
  4. Double-click: FruitNinjaCV.exe
  5. Game langsung terbuka, siap dimainkan!

  CATATAN: File .exe sudah mencakup semua assets
  (gambar + audio + model AI) di dalamnya.
  Tidak perlu install Python atau library apapun.

----------------------------------------------
  CARA 2: DARI SOURCE CODE (Untuk Developer)
----------------------------------------------

  Windows (otomatis install + jalankan):

    Double-click: setup_and_run.bat

  Mac / Linux:

    chmod +x setup_and_run.sh && ./setup_and_run.sh

  Manual (jika Python sudah terinstall):

    pip install mediapipe opencv-python pygame numpy
    python game.py

=================================================
  CARA BERMAIN
=================================================

Kontrol:

  Acungkan telunjuk    : Aktifkan kursor jari
  Tekuk jari lain      : Wajib ditekuk (tengah/manis/kelingking)
  Gerakkan telunjuk    : Tebas buah yang melayang
  Sentuh tombol START  : Mulai permainan dari menu

Keyboard:

  R    : Kembali ke menu utama
  Q    : Keluar game

Aturan:

  - Setiap buah berhasil ditebas = +1 score
  - Combo beruntun menambah bonus score
  - Menebas BOM = game over langsung
  - Buah lolos tanpa ditebas  = -1 nyawa
  - Game berakhir saat 3 nyawa habis

=================================================
  SCORING SYSTEM
=================================================

  Score per buah   : 1 + (combo // 3)
  Combo            : Bertambah setiap buah berhasil ditebas
  Combo reset      : Jika tidak ada tebasan selama 30 frame
  High Score       : Disimpan otomatis di highscore.txt

=================================================
  COMPUTER VISION PIPELINE
=================================================

  1. Kamera menangkap frame (640x480, 60 FPS)
  2. Frame dipadding 80px tiap sisi agar deteksi
     tetap akurat hingga ke tepi layar
  3. Frame dikirim ke MediaPipe Hand Landmarker
     secara async (LIVE_STREAM mode)
  4. Landmark tangan diekstrak (21 titik per tangan)
  5. is_index_raised() menentukan apakah telunjuk
     sedang terangkat berdasarkan perbandingan
     ekstensi antar jari
  6. Posisi ujung telunjuk (landmark 8) dikonversi
     ke koordinat layar
  7. FingerTracker memperhalus posisi dan menghitung
     kecepatan serta prediksi posisi berikutnya
  8. check_slice() mendeteksi apakah trail jari
     melewati area objek buah atau bom

=================================================
  PERFORMANCE INDICATOR
=================================================

  Speed Indicator  : Kecepatan gerakan jari (px/detik)
                     Hijau  = < 200 px/s  (lambat)
                     Kuning = 200-400 px/s (sedang)
                     Merah  = > 400 px/s  (cepat)
  Finger Status    : "Telunjuk AKTIF" / "Acungkan telunjuk"
  Trail Visual     : Jejak biru-cyan mengikuti gerakan jari
  Arrow Predictor  : Panah arah prediksi saat speed > 200

=================================================
  DEBUGGING
=================================================

Jika kamera tidak terdeteksi:

  - Pastikan tidak ada aplikasi lain yang memakai
    kamera (Zoom, Teams, OBS, dll)
  - Coba ganti index kamera: cv2.VideoCapture(1)

Jika jari tidak terdeteksi:

  - Pastikan pencahayaan cukup terang
  - Tekuk jari tengah, manis, kelingking dengan jelas
  - Jarak tangan ke kamera: 30-60 cm

Jika game lag:

  - Tutup aplikasi berat lainnya
  - Kurangi resolusi: ubah 640x480 menjadi 320x240

Jika .exe tidak bisa dibuka (Windows Defender):

  - Klik "More info" -> "Run anyway"
  - Ini terjadi karena .exe bukan dari publisher resmi

=================================================
  FUTURE IMPROVEMENTS
=================================================

[AI / CV]
- Deteksi gestur dua tangan sekaligus untuk efek
  tebasan ganda (double slash)
- Implementasi pose estimation untuk kontrol seluruh
  tubuh, bukan hanya jari
- Penambahan model klasifikasi gestur khusus
  (misal: gerakan menggenggam untuk pause game)

[GAMEPLAY]
- Mode multiplayer lokal (dua pemain bergantian)
- Level difficulty yang dinamis berdasarkan
  performa pemain (adaptive difficulty)
- Penambahan power-up: freeze time, extra life,
  dan score multiplier
- Leaderboard online untuk kompetisi antar pemain

[ANALYTICS]
- Dashboard statistik permainan: rata-rata score,
  win rate, buah paling sering lolos
- Visualisasi heatmap posisi tebasan pemain
- Grafik perkembangan score dari waktu ke waktu
- Ekspor data sesi bermain ke format CSV untuk
  analisis lebih lanjut

[ENGINEERING]
- Optimasi performa menggunakan threading terpisah
  untuk proses MediaPipe dan rendering frame
- Portabilitas ke platform Android / Raspberry Pi
- Packaging otomatis multi-platform (Windows + Mac)
  menggunakan GitHub Actions CI/CD

=================================================
  SYSTEM REQUIREMENTS
=================================================

Minimum:

  CPU      : Intel Core i3 / AMD Ryzen 3 (gen 8+)
  RAM      : 4 GB
  Python   : 3.9+
  Kamera   : Webcam 480p 30fps
  OS       : Windows 10, macOS 10.15, Ubuntu 20.04

Recommended:

  CPU      : Intel Core i5 / AMD Ryzen 5 (gen 10+)
  RAM      : 8 GB
  Python   : 3.11
  Kamera   : Webcam 720p 60fps
  OS       : Windows 11

=================================================
  DEPENDENCIES
=================================================

  mediapipe        >= 0.10.0
  opencv-python    >= 4.8.0
  pygame           >= 2.5.0
  numpy            >= 1.24.0

Install semua sekaligus:

  pip install mediapipe opencv-python pygame numpy

=================================================
  REFERENCES
=================================================

1. MediaPipe Hand Landmarker
   Google. (2023). Hand Landmark Detection Guide.
   https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

2. OpenCV Documentation
   OpenCV Team. (2023). OpenCV 4.x Documentation.
   https://docs.opencv.org/4.x/

3. Pygame Documentation
   Pygame Community. (2023). Pygame Documentation.
   https://www.pygame.org/docs/

4. Computer Vision: Algorithms and Applications
   Szeliski, R. (2022). 2nd Edition. Springer.
   https://szeliski.org/Book/

5. Real-Time Hand Gesture Recognition
   Zhang, F., et al. (2020). MediaPipe Hands:
   On-device Real-time Hand Tracking. arXiv:2006.10214
   https://arxiv.org/abs/2006.10214

=================================================
  LICENSE
=================================================

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files, to deal in the Software without
restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software.

This project uses MediaPipe by Google (Apache License 2.0)
and Pygame (LGPL License).

=================================================
  AUTHOR
=================================================

Project : Fruit Ninja CV - Computer Vision Game
Purpose : Tugas Akhir Semester (UAS) -   MACHINE LEARNING FOR INTELLIGENCE SYSTEM

Name    : Jordan Sebastian, Juan Felix's
Class   : 4PDS1
Lecturer: Eko Wahyu Prasetyo. S.T., M.Eng
Year    : 2026

=================================================
