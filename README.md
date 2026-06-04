# 🍉 Fruit Ninja CV — Hand Gesture Game with MediaPipe

> **Mata Kuliah:** Machine Learning for Intelligence System  
> **Universitas:** Universitas Bunda Mulia  
> **Authors:** Jordan Sebastian · Juan Felix

---

## 📌 Overview

**Fruit Ninja CV** adalah implementasi game Fruit Ninja berbasis **Computer Vision** yang dimainkan menggunakan gerakan jari telunjuk tangan secara real-time melalui webcam. Proyek ini memanfaatkan **MediaPipe Hand Landmarker** untuk mendeteksi dan melacak posisi jari, serta **OpenCV** sebagai engine rendering game.

Game ini merupakan demonstrasi aplikasi Machine Learning — khususnya **pose estimation** dan **gesture recognition** — dalam skenario interaktif secara real-time.

---

## 🎮 Demo

```
Acungkan telunjuk → Gerakkan tangan → Tebas buah!
```

| Status Jari | Aksi |
|---|---|
| ☝️ Telunjuk teracung (paling tinggi) | Kursor AKTIF |
| 🖐️ Semua jari terbuka rata | Kursor NONAKTIF |
| ✌️ Dua jari sama tinggi | Kursor NONAKTIF |

---

## 🧠 Machine Learning Component

### Model: MediaPipe Hand Landmarker

Model yang digunakan adalah **Hand Landmarker** dari Google MediaPipe, sebuah model deep learning yang dilatih untuk mendeteksi **21 titik landmark** pada tangan manusia secara real-time.

```
Landmark yang digunakan dalam proyek ini:
  [5]  MCP Telunjuk (pangkal)      [8]  Tip Telunjuk
  [9]  MCP Jari Tengah            [12] Tip Jari Tengah
  [13] MCP Jari Manis             [16] Tip Jari Manis
  [17] MCP Kelingking             [20] Tip Kelingking
```

### Algoritma Deteksi Gestur (Finger Extension)

Deteksi telunjuk menggunakan metode **Relative Finger Extension** — membandingkan seberapa jauh tip setiap jari berada di atas pangkalnya (MCP):

```python
def extension(tip_idx, mcp_idx):
    return hand[mcp_idx].y - hand[tip_idx].y  # positif = jari terangkat

# Telunjuk aktif jika ekstensinya paling besar dengan margin 0.03
is_active = (
    ext_index > 0 and
    ext_index > ext_middle + 0.03 and
    ext_index > ext_ring   + 0.03 and
    ext_index > ext_pinky  + 0.03
)
```

Pendekatan ini lebih robust dibanding threshold sederhana karena bekerja baik saat semua jari terbuka maupun saat hanya telunjuk yang terangkat.

---

## 🚀 Quick Start

### 1. Clone repository

```bash
git clone https://github.com/username/fruit-ninja-cv.git
cd fruit-ninja-cv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan game

```bash
python game.py
```

---

## 📋 Requirements

### System Requirements

| Komponen | Minimum | Recommended |
|---|---|---|
| **CPU** | Intel Core i3 / AMD Ryzen 3 | Intel Core i5 / AMD Ryzen 5 |
| **RAM** | 4 GB | 8 GB |
| **Python** | 3.8 | 3.10+ |
| **Webcam** | 640×480 @ 30fps | 1280×720 @ 60fps |
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

### Python Dependencies

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pygame>=2.5.0
```

---

## 📁 Directory Structure

```
fruit-ninja-cv/
│
├── main.py                    # Entry point utama
├── requirements.txt
├── README.md
├── highscore.txt              # Auto-generated saat bermain
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
└── hand_landmarker.task       # MediaPipe model (auto-download jika tidak ada)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│              main.py                    │
├──────────────┬──────────────────────────┤
│              │                          │
│   OpenCV     │     MediaPipe            │
│   (Render)   │     Hand Landmarker      │
│              │                          │
│  ┌─────────┐ │  ┌────────────────────┐  │
│  │ Game    │ │  │ is_index_raised()  │  │
│  │ Objects │ │  │ FingerTracker      │  │
│  │ Fruits  │ │  │ Smoothing          │  │
│  │ Bombs   │ │  │ Prediction         │  │
│  └────┬────┘ │  └────────┬───────────┘  │
│       │      │           │              │
│       └──────┴───────────┘              │
│              │                          │
│        check_slice()                    │
│        Score / Lives / Combo            │
└─────────────────────────────────────────┘
```

---

## 🎯 Game Features

- **Start Menu** — sentuh tombol START dengan telunjuk untuk mulai
- **Gesture Control** — kontrol penuh menggunakan gerakan telunjuk
- **Multi-directional Spawn** — buah muncul dari bawah, kiri, dan kanan layar
- **Combo System** — tebas beberapa buah berturut-turut untuk bonus poin
- **Bomb** — jangan tebas bom atau game over!
- **Sound Effects** — efek suara slice, throw, game over menggunakan Pygame
- **High Score** — skor tertinggi tersimpan otomatis
- **Screen Shake** — efek getaran layar saat mengenai bom
- **Trail Effect** — jejak visual mengikuti gerakan telunjuk

---

## 🕹️ Controls

| Input | Aksi |
|---|---|
| ☝️ Gerakkan telunjuk | Menebas buah |
| `R` | Kembali ke menu utama |
| `Q` | Keluar dari game |

---

## 🔧 How It Works

### Pipeline per Frame

```
Webcam Frame
    │
    ├─► cam_frame (asli) ──► MediaPipe Hand Landmarker
    │                              │
    │                         21 Landmarks
    │                              │
    │                     is_index_raised()
    │                              │
    │                     FingerTracker.update()
    │                              │
    │                     smooth() + predict()
    │                              │
    ├◄──────────────── finger_pos (x, y) ──────────────┤
    │                                                   │
    ├─► Apply Background Overlay                        │
    ├─► Update & Draw Game Objects                      │
    ├─► check_slice(obj, tracker, finger_pos) ◄─────────┘
    ├─► Draw Trail & Cursor
    ├─► Draw UI (Score, Lives, Combo)
    └─► imshow()
```

### Slice Detection (4 Metode)

Deteksi tebasan menggunakan 4 metode berlapis untuk memaksimalkan akurasi:

1. **Direct hit** — posisi telunjuk saat ini vs radius buah
2. **Trail points** — semua titik dalam histori pergerakan jari
3. **Line segment** — proyeksi garis antar titik trail ke pusat buah
4. **Predicted position** — posisi prediksi jari 18ms ke depan

---

## 📈 Future Improvements

**[AI]**
- Implementasi gesture recognition yang lebih kompleks (misal: kepalan tangan untuk pause)
- Deteksi dua tangan sekaligus untuk dual-player mode
- Model custom yang dilatih ulang untuk kondisi cahaya rendah

**[GAMEPLAY]**
- Level progression dengan difficulty yang meningkat
- Power-up system (freeze time, double score, dll.)
- Leaderboard online

**[ANALYTICS]**
- Grafik kecepatan gerakan tangan per sesi
- Heatmap area tebasan terbanyak
- Export statistik sesi ke CSV

**[ENGINEERING]**
- Build menjadi executable (.exe) menggunakan PyInstaller
- Optimasi performa dengan multiprocessing untuk deteksi dan render
- Dukungan kamera eksternal / IP camera

---

## 📚 References

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [Hand Landmark Detection — Google AI](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

---

## 📄 License

This project is created for educational purposes as part of the Machine Learning for Intelligence System course at Universitas Bunda Mulia. All game assets (images and sounds) used are credited to their respective owners.

---

## 👨‍💻 Authors

| Nama | NIM |
|---|---|
| Jordan Sebastian | 36240048 |
| Juan Felix | 36240051 |

> **Course:** [Machine Learning for Intelligence System]  
> **Program:** S1 Data Science — Universitas Bunda Mulia
