=================================================
FRUIT NINJA COMPUTER VISION INTERACTIVE GAME
=================================================

Version     : 1.0.0 (UAS Production)
Language    : Python 3.14+
Category    : Computer Vision / Human-Computer Interaction (HCI)
Framework   : MediaPipe Vision Tasks & OpenCV

=================================================
PROJECT OVERVIEW
=================================================

Project ini mengimplementasikan deteksi jari telunjuk
menggunakan MediaPipe Hand Landmarker untuk menciptakan
game Fruit Ninja interaktif berbasis Computer Vision. 
Pengguna dapat menebas buah yang muncul di layar secara 
real-time hanya dengan menggerakkan jari telunjuk di depan 
kamera tanpa alat kendali tambahan.

Fitur Utama yang tersedia:

1. Interactive Main Menu
   - Layar start interaktif menggunakan deteksi koordinat.
   - Game baru akan dimulai jika jari menduduki tombol START.

2. Precision Index Finger Tracking
   - Sistem klasifikasi jari berbasis jarak Landmark MCP & TIP.
   - Mengunci pelacakan eksklusif hanya untuk jari telunjuk.

3. Zero-Latency Buffer Management
   - Optimalisasi kamera menggunakan pengosongan buffer berkala.
   - Menghilangkan jeda input (input lag) antara gerakan asli dan kursor.

4. Dynamic Velocity Smoothing
   - Algoritma interpolasi posisi berdasarkan kalkulasi kecepatan tebasan.

5. Interactive Audio Feedback
   - Integrasi sound effect non-blocking menggunakan Pygame Mixer.

=================================================
GAME OBJECTIVE
=================================================

Tujuan pemain dalam Fruit Ninja CV adalah:

- Menebas setiap buah yang muncul di layar untuk mencetak poin.
- Menghindari tebasan pada objek BOM yang memicu Game Over.
- Mempertahankan 3 sisa nyawa (Lives) agar tidak gugur.
- Membangun tebasan beruntun (Combo System) untuk melipatgandakan skor.
- Memecahkan rekor skor tertinggi (High Score) yang tersimpan otomatis.

=================================================
PROJECT ARCHITECTURE
=================================================

+-----------------------+
|        game.py        |
+-----------+-----------+
            |
            v
+-----------------------+
|   MediaPipe Vision    | (Hand Landmarker Async)
+-----------+-----------+
            |
            v
+-----------------------+
|    FingerTracker      | (Smoothing & Prediction)
+-----------+-----------+
            |
    +-------+-------+
    |               |
    v               v
Slice Logic     Pygame Mixer
 (OpenCV)       (Sound FX)
    |               |
    +-------+-------+
            |
            v
     Game Environment

=================================================
DIRECTORY STRUCTURE
=================================================

##Actual Structure##

Fruit_Ninja_UAS
│
├── assets/
│   ├── apple.png
│   ├── bomb.png
│   ├── Clean-Slice-1.mp3
│   ├── Game-over.mp3
│   ├── Game-start.mp3
│   ├── Throw-bomb.mp3
│   └── Throw-fruit.mp3
│
├── dist/
│   └── Fruit_Ninja_UAS/
│       └── Fruit_Ninja_UAS.exe
│
├── game.py
├── build.spec
├── hand_landmarker.task
├── highscore.txt
└── README.md

=================================================
QUICK START
=================================================

Menjalankan game lewat Source Code:

python game.py

Menjalankan game lewat Executable (.exe):

1. Buka folder dist/Fruit_Ninja_UAS/
2. Klik ganda pada file Fruit_Ninja_UAS.exe

*Catatan: Pastikan folder 'assets' dan file 'hand_landmarker.task' 
berada di tempat yang sejajar dengan file .exe Anda.

=================================================
GAME CONTROLS & FEATURES
=================================================

Mulai Permainan (Layar Utama):
Arahkan kursor lingkaran jari telunjuk Anda tepat ke dalam 
lingkaran tombol hijau bertuliskan "START" di tengah layar.

Menebas Objek (Gameplay):
Acungkan HANYA jari telunjuk Anda (tekuk jari lainnya) ke arah 
kamera, lalu lakukan gerakan mengayun cepat memotong buah.

Kembali ke Menu Utama / Restart:
Tekan tombol 'r' pada keyboard saat permainan berlangsung 
atau saat Anda berada di layar Game Over.

Keluar dari Game:
Tekan tombol 'q' pada keyboard kapan saja.

=================================================
SYSTEM OPTIMIZATION & EVALUATION
=================================================

Untuk memastikan game berjalan 60 FPS tanpa lag, sistem 
menggunakan 3 pilar optimasi matematis:

1. Real-time Camera Buffer Overwrite
   Mengunci properti `CAP_PROP_BUFFERSIZE` bernilai 1. Menghapus 
   antrean frame lama pada RAM Windows yang biasa memicu lag visual.

2. Adaptive Frame Padding ($PAD = 80$)
   Menambahkan border tiruan di sekeliling matriks kamera sebelum 
   diumpan ke AI. Mencegah kegagalan tracking saat telunjuk berada 
   di tepi layar.

3. Hitbox Expansion ($Radius + 40$)
   Memperlebar area intersep deteksi tebasan menggunakan perhitungan 
   garis interpolasi linear jarak euclidean antara dua titik frame.

=================================================
DEBUGGING
=================================================

Menampilkan Indikator Status di Layar:

- "Telunjuk AKTIF" (Hijau): AI berhasil memvalidasi postur jari telunjuk.
- "Acungkan telunjuk" (Oranye): AI mendeteksi tangan namun postur salah/mengepal.

Pengecekan Log Konsol:
Jika aset gambar atau audio hilang, konsol akan memunculkan pesan:
"Warning: asset 'nama_file' tidak ditemukan di folder assets."

=================================================
BUILD CONFIGURATION
=================================================

Project ini di-compile menjadi standalone architecture menggunakan PyInstaller.

Perintah untuk membangun ulang bundle (.exe):

pyinstaller build.spec --clean

PENTING (Solusi Windows Smart App Control):
Karena aplikasi ini dibangun secara lokal, Windows Smart App Control 
mungkin akan memblokir file .exe saat pertama kali dijalankan dari internet.

Cara Mengatasi:
1. Klik kanan pada file `Fruit_Ninja_UAS.exe`
2. Pilih menu 'Properties'
3. Pada tab 'General' bagian bawah, centang opsi kotak 'Unblock'
4. Klik 'Apply' lalu 'OK'. Game bisa langsung dimainkan.

=================================================
COMPUTER VISION & SMOOTHING CONCEPT
=================================================

Untuk menstabilkan pergerakan kursor dari guncangan (jittering) 
kamera tanpa mengurangi responsivitas kecepatan, pelacakan posisi 
menggunakan persamaan Exponential Moving Average (EMA) Dinamis:
$$S_x = X_{raw} \cdot \alpha + last_x \cdot (1 - \alpha)$$

Dimana:

Sx      = Koordinat kursor hasil smoothing yang dirender ke layar
Xraw    = Koordinat mentah (raw coordinate) ujung telunjuk dari MediaPipe
last_x  = Koordinat kursor pada satu frame sebelumnya
α       = Faktor bobot responsivitas (Smoothing Factor)

Nilai Alpha (α) ditentukan secara dinamis berdasarkan kecepatan objek:
- Jika Speed > 150 px/s  -> α = 1.00 (Kursor menempel instan 100% tanpa delay)
- Jika Speed > 80 px/s   -> α = 0.95 (Smoothing sangat tipis)
- Jika Speed <= 80 px/s  -> α = 0.85 (Smoothing meredam jitter tangan bergetar)

=================================================
FUTURE IMPROVEMENTS
=================================================

Beberapa pengembangan yang direkomendasikan:

[AI]
- Menambahkan arsitektur Multi-Hand Tracking untuk mendukung mode permainan dua pemain (Co-Op Mode).
- Mengintegrasikan deteksi Gesture khusus untuk memicu jurus spesial (misalnya mengepalkan tangan untuk membekukan waktu).

[GAMEPLAY]
- Menambahkan variasi buah spesial (seperti Pisang Frenzy atau Buah Naga Bonus).
- Mengimplementasikan sistem partikel potongan daging buah belah dua (Sliced Mesh Splitting) yang lebih realistis.

[ANALYTICS]
- Menyimpan histori data tebasan tercepat dan akurasi rasio hit/miss ke dalam file lokal untuk analisis statistik pemain.

[ENGINEERING]
- Migrasi sistem rendering UI sepenuhnya ke Pygame Graphics Framework untuk meringankan beban kerja CPU.

=================================================
SYSTEM REQUIREMENTS
=================================================

Minimum:

CPU      : Intel Core i3 / AMD Ryzen 3 (Dual-Core 2.4 GHz)
RAM      : 4 GB
Kamera   : WebCam bawaan laptop (30 FPS)
Python   : Python 3.10 atau lebih tinggi

Recommended:

CPU      : Intel Core i5 / AMD Ryzen 5 (Quad-Core 3.0 GHz)
RAM      : 8 GB atau lebih tinggi
Kamera   : External HD WebCam (60 FPS, Pencahayaan Bagus)
Python   : Python 3.14.x

=================================================
REFERENCES
=================================================

- Google MediaPipe Solutions: Hand Landmarker Guide for Python LiveStream Mode.
- OpenCV-Python Documentation: VideoCapture properties and Matrix transformations.
- Pygame Mixer Module Documentation: Asynchronous Sound Loading & Channel Management.

=================================================
LICENSE
=================================================

Project ini dilisensikan di bawah MIT License - bebas digunakan untuk keperluan edukasi dan akademik.

=================================================
AUTHOR
=================================================

Project : Fruit Ninja Computer Vision - Standalone Interactive Game
Purpose : Project Implementasi Tugas Akhir / Ujian Akhir Semester (UAS) Mata Kuliah Teknologi Kecerdasan Buatan / Data Science
Nama    : Jordan Sebastian , Juan Felix's
=================================================
