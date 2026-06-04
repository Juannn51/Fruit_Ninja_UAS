# build.spec
# File konfigurasi PyInstaller untuk Fruit Ninja UAS - Mode ONE FILE

import os
import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE

# Path root project
ROOT = os.path.abspath(".")

# OTOMATISASI: Cari letak asli libmediapipe.dll
import mediapipe
mediapipe_dir = os.path.dirname(mediapipe.__file__)
mediapipe_dll = os.path.join(mediapipe_dir, "tasks", "c", "libmediapipe.dll")

mediapipe_binaries = []
if os.path.exists(mediapipe_dll):
    mediapipe_binaries.append((mediapipe_dll, os.path.join("mediapipe", "tasks", "c")))

a = Analysis(
    ["game.py"],
    pathex=[ROOT],
    binaries=mediapipe_binaries,
    datas=[
        # Masukkan assets dan model ke dalam bungkusan .exe
        ("assets",  "assets"),                  
        ("hand_landmarker.task", "."),          
    ],
    hiddenimports=[
        "mediapipe",
        "mediapipe.tasks.c",                    
        "cv2",
        "pygame",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Fruit_Ninja_UAS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                      # False = Sembunyikan terminal hitam
)
# Perhatikan: Tidak ada blok COLLECT di sini karena kita membungkusnya jadi satu file