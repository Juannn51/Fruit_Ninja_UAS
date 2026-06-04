import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
import os
import math
import time
from collections import deque
import sys

# Menyembunyikan teks sambutan pygame di terminal
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# ==========================================
# PERBAIKAN PATH UNTUK MODE ONE-FILE .EXE
# ==========================================
def get_base_path():
    """Mencari folder temporary asset saat dijalankan sebagai .exe tunggal"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_exe_path():
    """Mencari folder tempat file .exe berada (untuk simpan highscore agar tidak hilang)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
EXE_DIR = get_exe_path()

# ============ SETUP MEDIAPIPE ============
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')
if not os.path.exists(model_path):
    print("Downloading hand model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

latest_result = None
latest_timestamp = 0

def update_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, latest_timestamp
    latest_result = result
    latest_timestamp = timestamp_ms

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result
)

landmarker = HandLandmarker.create_from_options(options)

# ============ SETUP KAMERA ============
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_CONTRAST, 50)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# ============ WINDOW ============
WINDOW_NAME = 'Fruit Ninja - Enhanced Tracking'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)   
cv2.resizeWindow(WINDOW_NAME, 900, 600)          

# ============ LOAD ASSETS (GAMBAR) ============
assets_path = os.path.join(BASE_DIR, 'assets')

fruit_names = ['apple', 'orange', 'watermelon', 'kiwi', 'strawberry', 'peach']
fruit_colors = {
    'apple': (0, 0, 255),
    'orange': (0, 140, 255),
    'watermelon': (0, 180, 0),
    'kiwi': (100, 200, 0),
    'strawberry': (0, 0, 200),
    'peach': (0, 100, 255)
}

fruit_images = {}
for name in fruit_names:
    img_path = os.path.join(assets_path, f'{name}.png')
    if os.path.exists(img_path):
        fruit_images[name] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    else:
        fruit_images[name] = None
        print(f"Warning: asset '{name}.png' tidak ditemukan di folder assets.")

bomb_img = None
bomb_path = os.path.join(assets_path, 'bomb.png')
if os.path.exists(bomb_path):
    bomb_img = cv2.imread(bomb_path, cv2.IMREAD_UNCHANGED)
else:
    print("Warning: asset 'bomb.png' tidak ditemukan di folder assets.")

# ============ LOAD ASSETS (AUDIO) ============
pygame.mixer.init()
sounds = {}
sound_names = ['Clean-Slice-1', 'Game-over', 'Game-start', 'Throw-bomb', 'Throw-fruit']

for s_name in sound_names:
    mp3_path = os.path.join(assets_path, f'{s_name}.mp3')
    wav_path = os.path.join(assets_path, f'{s_name}.wav')
    
    if os.path.exists(mp3_path):
        sounds[s_name] = pygame.mixer.Sound(mp3_path)
    elif os.path.exists(wav_path):
        sounds[s_name] = pygame.mixer.Sound(wav_path)
    else:
        sounds[s_name] = None
        print(f"Warning: Audio '{s_name}' (.mp3/.wav) tidak ditemukan di folder assets.")

def play_sound(name):
    if name in sounds and sounds[name] is not None:
        sounds[name].play()


# ============ DETEKSI TELUNJUK ============
def is_index_raised(hand):
    def ext(tip, mcp):
        return hand[mcp].y - hand[tip].y 

    e_index  = ext(8,  5)
    e_middle = ext(12, 9)
    e_ring   = ext(16, 13)
    e_pinky  = ext(20, 17)

    if e_index <= 0:
        return False

    MARGIN = 0.01
    return (e_index > e_middle + MARGIN and
            e_index > e_ring   + MARGIN and
            e_index > e_pinky  + MARGIN)


# ============ FINGER TRACKER ============
class FingerTracker:
    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)
        self.velocity = (0, 0)
        self.speed = 0
        self.direction = 0
        self.predicted_pos = None

    def update(self, x, y):
        now = time.time()
        self.history.append((x, y, now))
        if len(self.history) >= 2:
            x1, y1, t1 = self.history[-2]
            x2, y2, t2 = self.history[-1]
            dt = max(t2 - t1, 0.001)
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
            self.velocity = (vx, vy)
            self.speed = math.sqrt(vx**2 + vy**2)
            self.direction = math.degrees(math.atan2(vy, vx))
            predict_time = 0.018
            self.predicted_pos = (
                int(x2 + vx * predict_time),
                int(y2 + vy * predict_time)
            )

    def smooth(self, raw_x, raw_y):
        if not self.history:
            return raw_x, raw_y
        last_x, last_y, _ = self.history[-1]
        
        if self.speed > 150:
            factor = 1.0  
        elif self.speed > 80:
            factor = 0.95
        else:
            factor = 0.85 
            
        sx = int(raw_x * factor + last_x * (1 - factor))
        sy = int(raw_y * factor + last_y * (1 - factor))
        return sx, sy

    def get_slice_points(self):
        return [(int(x), int(y)) for x, y, _ in self.history]

    def reset(self):
        self.history.clear()
        self.velocity = (0, 0)
        self.speed = 0
        self.predicted_pos = None


# ============ SPAWN HELPER ============
def spawn_fruit(w, h, score):
    roll = random.random()
    if roll < 0.60:
        direction = 'bottom'
    elif roll < 0.80:
        direction = 'left'
    else:
        direction = 'right'

    base_vy = random.uniform(-16, -13)

    if direction == 'bottom':
        x  = random.randint(100, w - 100)
        y  = h + 50
        vx = random.uniform(-4, 4)
        vy = base_vy
    elif direction == 'left':
        x  = -50
        y  = random.randint(int(h * 0.25), int(h * 0.5))
        vx = random.uniform(7, 12)
        vy = random.uniform(-10, -6)
    else:
        x  = w + 50
        y  = random.randint(int(h * 0.25), int(h * 0.5))
        vx = random.uniform(-12, -7)
        vy = random.uniform(-10, -6)

    return x, y, vx, vy


# ============ GAME OBJECT ============
class GameObject:
    def __init__(self, x, y, vx, vy, obj_type='fruit', fruit_type='apple'):
        self.x    = x
        self.y    = y
        self.vx   = vx
        self.vy   = vy
        self.type = obj_type
        self.fruit_type     = fruit_type
        self.radius         = 35 if obj_type == 'fruit' else 30
        self.sliced         = False
        self.particles      = []
        self.rotation       = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-6, 6)
        self.slice_cooldown = 0

    def update(self):
        if not self.sliced:
            self.x  += self.vx
            self.vy += 0.35
            self.y  += self.vy
            self.rotation += self.rotation_speed
        if self.slice_cooldown > 0:
            self.slice_cooldown -= 1

    def slice(self):
        if self.slice_cooldown > 0:
            return False
        self.sliced = True
        self.slice_cooldown = 10
        for _ in range(18):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 14)
            self.particles.append({
                'x': self.x, 'y': self.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 4,
                'life': random.randint(18, 35),
                'color': fruit_colors.get(self.fruit_type, (255, 255, 255))
                         if self.type == 'fruit' else (50, 50, 50)
            })
        return True

    def update_particles(self):
        for p in self.particles:
            p['x']  += p['vx']
            p['y']  += p['vy']
            p['vy'] += 0.4
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def draw(self, frame):
        if self.sliced:
            for p in self.particles:
                alpha = p['life'] / 35
                color = tuple(int(c * alpha) for c in p['color'])
                size  = max(1, int(4 + p['life'] / 3))
                cv2.circle(frame, (int(p['x']), int(p['y'])), size, color, -1)
            return

        x, y = int(self.x), int(self.y)
        if self.type == 'bomb':
            if bomb_img is not None:
                self._draw_image(frame, bomb_img, x, y, self.radius * 2)
            else:
                cv2.circle(frame, (x, y), self.radius, (30, 30, 30), -1)
                cv2.circle(frame, (x, y), self.radius, (200, 200, 200), 2)
                cv2.line(frame, (x, y - 10), (x - 5, y - 20), (200, 200, 200), 2)
                cv2.circle(frame, (x - 5, y - 22), 4, (0, 0, 255), -1)
        else:
            fruit_img = fruit_images.get(self.fruit_type)
            if fruit_img is not None:
                self._draw_image(frame, fruit_img, x, y, self.radius * 2)
            else:
                color = fruit_colors.get(self.fruit_type, (255, 0, 0))
                cv2.circle(frame, (x + 2, y + 2), self.radius, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), self.radius, color, -1)
                cv2.circle(frame, (x, y), self.radius, (255, 255, 255), 2)
                cv2.circle(frame, (x - 8, y - 8), self.radius // 3, (255, 255, 255), -1)

    def _draw_image(self, frame, img, x, y, size):
        try:
            img_resized = cv2.resize(img, (size, size))
            fh, fw = frame.shape[:2]
            ih, iw = img_resized.shape[:2]
            x1 = max(0, x - iw // 2); y1 = max(0, y - ih // 2)
            x2 = min(fw, x + iw // 2); y2 = min(fh, y + ih // 2)
            ix1 = x1 - (x - iw // 2); iy1 = y1 - (y - ih // 2)
            ix2 = ix1 + (x2 - x1);    iy2 = iy1 + (y2 - y1)
            if x2 <= x1 or y2 <= y1:
                return
            if img_resized.shape[2] == 4:
                alpha = img_resized[iy1:iy2, ix1:ix2, 3] / 255.0
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        frame[y1:y2, x1:x2, c] * (1 - alpha) +
                        img_resized[iy1:iy2, ix1:ix2, c] * alpha
                    )
            else:
                frame[y1:y2, x1:x2] = img_resized[iy1:iy2, ix1:ix2]
        except:
            pass

    def offscreen(self, h, w):
        return (self.y > h + 120 or self.x < -120 or self.x > w + 120
                or (self.sliced and len(self.particles) == 0))


# ============ SLICE DETECTION ============
def check_slice(obj, tracker, finger_pos):
    if obj.sliced or obj.slice_cooldown > 0:
        return False

    HIT = obj.radius + 40 

    if finger_pos:
        dx = finger_pos[0] - obj.x
        dy = finger_pos[1] - obj.y
        if math.sqrt(dx**2 + dy**2) < HIT:
            return True

    points = tracker.get_slice_points()
    for px, py in points:
        if math.sqrt((px - obj.x)**2 + (py - obj.y)**2) < HIT:
            return True

    if len(points) >= 2:
        for i in range(len(points) - 1):
            p1 = points[i]; p2 = points[i + 1]
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
            denom = dx*dx + dy*dy
            t = max(0.0, min(1.0, ((obj.x - p1[0])*dx + (obj.y - p1[1])*dy) / denom))
            cx = p1[0] + t*dx; cy = p1[1] + t*dy
            if math.sqrt((obj.x - cx)**2 + (obj.y - cy)**2) < HIT - 4:
                return True

    if tracker.predicted_pos:
        px, py = tracker.predicted_pos
        if math.sqrt((px - obj.x)**2 + (py - obj.y)**2) < HIT - 4:
            return True

    return False


# ============ GAME STATE ============
fruits         = []
score          = 0
high_score     = 0
lives          = 3
spawn_timer    = 0
finger_pos     = None
trail          = deque(maxlen=30)
game_over      = False
game_started   = False 
combo          = 0
combo_timer    = 0
shake_timer    = 0
finger_tracker = FingerTracker(history_size=10)
index_was_up   = False   

# Perbaikan path highscore agar disimpan di folder .exe, bukan di ram virtual
high_score_file = os.path.join(EXE_DIR, 'highscore.txt')
try:
    with open(high_score_file, 'r') as f:
        high_score = int(f.read())
except:
    high_score = 0

print("=" * 55)
print("🍉 FRUIT NINJA CV 🍉")
print("=" * 55)
print("Cara main:")
print("- Acungkan HANYA telunjuk (jari lain ditekuk)")
print("- Sentuh tombol START di layar untuk mulai")
print("- Gerakkan telunjuk untuk menebas buah")
print("- Jangan tebas BOM!")
print("Tekan: 'q' keluar | 'r' kembali ke menu")
print("=" * 55)


# ============ MAIN LOOP ============
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if shake_timer > 0:
        shake_timer -= 1
        M = np.float32([[1, 0, random.randint(-8, 8)], [0, 1, random.randint(-8, 8)]])
        frame = cv2.warpAffine(frame, M, (w, h))

    overlay = frame.copy()
    cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # ===== DETEKSI TANGAN =====
    PAD = 80  
    frame_padded = cv2.copyMakeBorder(frame, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    pad_h, pad_w = frame_padded.shape[:2]

    rgb      = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, int(time.time() * 1000))

    finger_detected = False
    if latest_result and latest_result.hand_landmarks:
        hand = latest_result.hand_landmarks[0]
        index_up = is_index_raised(hand)

        if index_up:
            tip = hand[8]
            raw_x = int(tip.x * pad_w) - PAD
            raw_y = int(tip.y * pad_h) - PAD
            raw_x = max(0, min(w - 1, raw_x))
            raw_y = max(0, min(h - 1, raw_y))

            if not index_was_up:
                finger_tracker.reset()

            finger_tracker.update(raw_x, raw_y)
            sx, sy     = finger_tracker.smooth(raw_x, raw_y)
            finger_pos = (sx, sy)
            trail.append(finger_pos)
            finger_detected = True
            index_was_up = True
        else:
            index_was_up = False
            finger_pos   = None
            finger_tracker.reset()
            if trail:
                trail.popleft()
    else:
        index_was_up = False
        finger_pos   = None
        finger_tracker.reset()
        if trail:
            trail.popleft()


    # ============ LOGIKA START MENU ============
    if not game_started:
        # Gambar Judul
        cv2.putText(frame, "FRUIT NINJA CV", (w//2 - 180, h//2 - 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "Gunakan telunjuk untuk menyentuh tombol", (w//2 - 170, h//2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Gambar Tombol START
        btn_x, btn_y = w // 2, h // 2 + 30
        btn_r = 50
        cv2.circle(frame, (btn_x, btn_y), btn_r, (0, 180, 0), -1)
        cv2.circle(frame, (btn_x, btn_y), btn_r, (255, 255, 255), 3)
        cv2.putText(frame, "START", (btn_x - 38, btn_y + 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Deteksi Sentuhan Telunjuk ke Tombol START
        if finger_pos:
            fx, fy = finger_pos
            if math.sqrt((fx - btn_x)**2 + (fy - btn_y)**2) < btn_r:
                game_started = True
                game_over = False
                fruits.clear()
                score = 0
                lives = 3
                combo = 0
                combo_timer = 0
                spawn_timer = 0
                trail.clear()
                finger_tracker.reset()
                play_sound('Game-start')
    
    # ============ LOGIKA GAMEPLAY ============
    else:
        # --- SPAWN ---
        if not game_over:
            spawn_timer += 1
            delay = max(18, 45 - score // 2)
            if spawn_timer > delay:
                bomb_chance = min(0.35, 0.1 + score * 0.005)
                x, y, vx, vy = spawn_fruit(w, h, score)
                
                if random.random() < bomb_chance:
                    fruits.append(GameObject(x, y, vx, vy, 'bomb'))
                    play_sound('Throw-bomb')
                else:
                    fruits.append(GameObject(x, y, vx, vy, 'fruit', random.choice(fruit_names)))
                    play_sound('Throw-fruit')
                    
                if random.random() < 0.30 and score > 2:
                    x2, y2, vx2, vy2 = spawn_fruit(w, h, score)
                    fruits.append(GameObject(x2, y2, vx2, vy2, 'fruit', random.choice(fruit_names)))
                    play_sound('Throw-fruit')
                spawn_timer = 0

        # --- UPDATE & DRAW OBJECTS ---
        for obj in fruits[:]:
            obj.update()
            if obj.sliced:
                obj.update_particles()

            if not obj.sliced and not game_over:
                if check_slice(obj, finger_tracker, finger_pos):
                    if obj.slice():
                        if obj.type == 'bomb':
                            if not game_over:
                                game_over   = True
                                shake_timer = 20
                                play_sound('Game-over')
                                cv2.putText(frame, "BOOM!", (w//2-60, h//2),
                                            cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5)
                        else:
                            play_sound('Clean-Slice-1')
                            combo      += 1
                            combo_timer = 30
                            score      += 1 + combo // 3

            obj.draw(frame)

            if obj.offscreen(h, w):
                try:
                    fruits.remove(obj)
                except ValueError:
                    pass
                if obj.type == 'fruit' and not obj.sliced:
                    lives -= 1
                    combo  = 0
                    if lives <= 0 and not game_over:
                        game_over = True
                        play_sound('Game-over')

        if combo_timer > 0:
            combo_timer -= 1
            if combo_timer == 0:
                combo = 0

        # --- UI GAMEPLAY ---
        cv2.putText(frame, f"Score: {score}",      (15, 35), cv2.FONT_HERSHEY_DUPLEX,  1,   (255,255,255), 2)
        cv2.putText(frame, f"Best:  {high_score}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        spd = min(int(finger_tracker.speed), 999)
        sc  = (0,255,0) if spd < 200 else (0,255,255) if spd < 400 else (0,100,255)
        cv2.putText(frame, f"Speed: {spd}", (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 1)

        for i in range(3):
            hx = w - 120 + i * 35
            lc = (0,0,255) if i >= lives else (80,80,80)
            lt = 3          if i >= lives else 2
            cv2.putText(frame, "X", (hx, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, lc, lt)

        if combo >= 3:
            cv2.putText(frame, f"COMBO x{combo}!", (w//2-80, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,255,255), 3)

        if game_over:
            go_ov = frame.copy()
            cv2.rectangle(go_ov, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(go_ov, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "GAME OVER",             (w//2-130, h//2-30),  cv2.FONT_HERSHEY_DUPLEX,  2,    (0,0,255),     4)
            cv2.putText(frame, f"Score: {score}",       (w//2-80,  h//2+25),  cv2.FONT_HERSHEY_DUPLEX,  1.2,  (255,255,255), 2)
            cv2.putText(frame, "Tekan 'r' kembali ke Menu", (w//2-135, h//2+70),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)
            cv2.putText(frame, "atau 'q' keluar",       (w//2-90,  h//2+105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150,150,150), 2)
            if score > high_score:
                high_score = score
                try:
                    with open(high_score_file, 'w') as f:
                        f.write(str(high_score))
                except:
                    pass


    # ============ DRAW TRAIL & KURSOR (DITAMPILKAN DI SEMUA STATE) ============
    if len(trail) >= 2:
        tlist = list(trail)
        for i in range(1, len(tlist)):
            a  = i / len(tlist)
            th = max(1, int(2 + a * 10))
            c  = (int(50*(1-a)), int(200*a), int(255*a))
            cv2.line(frame, tlist[i-1], tlist[i], c, th)

    if finger_pos:
        oc = (0, 100, 255) if finger_tracker.speed > 400 else (100, 100, 255)
        ic = (0, 200, 255) if finger_tracker.speed > 400 else (150, 200, 255)
        cv2.circle(frame, finger_pos, 24, oc, 2)
        cv2.circle(frame, finger_pos, 18, ic, 2)
        cv2.circle(frame, finger_pos,  8, (255, 255, 255), -1)
        cv2.circle(frame, finger_pos,  4, (0, 255, 255), -1)
        if finger_tracker.speed > 200 and finger_tracker.predicted_pos:
            px = max(0, min(w-1, finger_tracker.predicted_pos[0]))
            py = max(0, min(h-1, finger_tracker.predicted_pos[1]))
            cv2.arrowedLine(frame, finger_pos, (px, py), (0, 255, 200), 2, tipLength=0.4)

    if finger_detected:
        status_color = (0, 255, 100)
        status_text  = "Telunjuk AKTIF"
    else:
        status_color = (0, 100, 200)
        status_text  = "Acungkan telunjuk"
    cv2.putText(frame, status_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

    # Info tombol exit/restart kecil di bawah
    cv2.putText(frame, "'r' restart/menu | 'q' quit", (w//2-110, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # KEMBALI KE MENU
        game_started = False
        game_over = False  
        fruits.clear()
        trail.clear()
        finger_tracker = FingerTracker(history_size=10)
        index_was_up   = False

# Cleanup
landmarker.close()
cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
print(f"\nFinal Score: {score} | High Score: {high_score}")
print("Terima kasih sudah bermain!")