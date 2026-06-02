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

# ============ SETUP MEDIAPIPE ============
model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
cap.set(cv2.CAP_PROP_CONTRAST, 50)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# ============ LOAD ASSETS ============
assets_path = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(assets_path, exist_ok=True)

# --- AUTO DOWNLOAD GAMBAR BUAH (hanya jika belum ada) ---
import urllib.request as _url

fruit_urls = {
    'apple':      'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/1F34E.png',
    'orange':     'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/1F34A.png',
    'watermelon': 'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/1F349.png',
    'kiwi':       'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/1F95D.png',
    'strawberry': 'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/1F353.png',
    'peach':      'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/1F351.png',
    'bomb':       'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/1F4A3.png',
}

def download_assets():
    all_ok = True
    for name, url in fruit_urls.items():
        path = os.path.join(assets_path, f'{name}.png')
        if not os.path.exists(path):
            try:
                print(f'  Downloading {name}.png ...', end=' ', flush=True)
                _url.urlretrieve(url, path)
                print('OK')
            except Exception as e:
                print(f'GAGAL ({e})')
                all_ok = False
    return all_ok

print('Mengecek asset gambar buah...')
if download_assets():
    print('Semua asset siap!\n')
else:
    print('Beberapa asset gagal, akan pakai lingkaran sebagai pengganti.\n')

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

bomb_img = None
bomb_path = os.path.join(assets_path, 'bomb.png')
if os.path.exists(bomb_path):
    bomb_img = cv2.imread(bomb_path, cv2.IMREAD_UNCHANGED)

# ============ FINGER TRACKER ============
class FingerTracker:
    def __init__(self, history_size=8):
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
            predict_time = 0.016
            self.predicted_pos = (
                int(x2 + vx * predict_time),
                int(y2 + vy * predict_time)
            )

    def smooth(self, raw_x, raw_y):
        if not self.history:
            return raw_x, raw_y
        last_x, last_y, _ = self.history[-1]
        if self.speed > 500:
            factor = 0.98
        elif self.speed > 300:
            factor = 0.92
        elif self.speed > 150:
            factor = 0.80
        else:
            factor = 0.65
        sx = int(raw_x * factor + last_x * (1 - factor))
        sy = int(raw_y * factor + last_y * (1 - factor))
        return sx, sy

    def get_slice_points(self):
        return [(int(x), int(y)) for x, y, _ in self.history]

# ============ SPAWN HELPER ============
def spawn_fruit(w, h, score):
    """
    3 jenis arah spawn:
      'bottom' - meloncat dari bawah layar (seperti biasa)
      'left'   - meluncur dari sisi kiri
      'right'  - meluncur dari sisi kanan
    """
    # Probabilitas arah: 60% bawah, 20% kiri, 20% kanan
    roll = random.random()
    if roll < 0.60:
        direction = 'bottom'
    elif roll < 0.80:
        direction = 'left'
    else:
        direction = 'right'

    # Kecepatan vertikal (makin tinggi negatif = makin tinggi loncatan)
    # Dinaikkan dari -14~-9 menjadi -20~-15
    # vy dibatasi agar puncak loncatan tidak keluar atas layar
    # Rumus puncak: y_puncak = y_awal + vy^2 / (2 * gravitasi)
    # gravitasi = 0.35, layar h=480 → vy maks ≈ -13 agar puncak ~y=50
    base_vy = random.uniform(-13, -10)

    if direction == 'bottom':
        x  = random.randint(80, w - 80)
        y  = h + 50
        vx = random.uniform(-5, 5)
        vy = base_vy

    elif direction == 'left':
        x  = -50
        y  = random.randint(int(h * 0.5), int(h * 0.85))
        vx = random.uniform(6, 10)
        vy = random.uniform(-12, -9)   # Lebih rendah agar tidak keluar atas

    else:  # right
        x  = w + 50
        y  = random.randint(int(h * 0.5), int(h * 0.85))
        vx = random.uniform(-10, -6)
        vy = random.uniform(-12, -9)

    return x, y, vx, vy


# ============ GAME OBJECT ============
class GameObject:
    def __init__(self, x, y, vx, vy, obj_type='fruit', fruit_type='apple'):
        self.x    = x
        self.y    = y
        self.vx   = vx
        self.vy   = vy
        self.type = obj_type
        self.fruit_type    = fruit_type
        self.radius        = 35 if obj_type == 'fruit' else 30
        self.sliced        = False
        self.particles     = []
        self.rotation      = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-6, 6)
        self.slice_cooldown = 0

    def update(self):
        if not self.sliced:
            self.x  += self.vx
            self.vy += 0.35        # Gravitasi
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
            p['x']   += p['vx']
            p['y']   += p['vy']
            p['vy']  += 0.4
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
            fh, fw     = frame.shape[:2]
            ih, iw     = img_resized.shape[:2]
            x1 = max(0, x - iw // 2);  y1 = max(0, y - ih // 2)
            x2 = min(fw, x + iw // 2); y2 = min(fh, y + ih // 2)
            ix1 = x1 - (x - iw // 2);  iy1 = y1 - (y - ih // 2)
            ix2 = ix1 + (x2 - x1);     iy2 = iy1 + (y2 - y1)
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
        # Buah dari kiri/kanan juga keluar jika melewati sisi berlawanan
        out_bottom = self.y > h + 120
        out_left   = self.x < -120
        out_right  = self.x > w + 120
        done_particles = self.sliced and len(self.particles) == 0
        return out_bottom or out_left or out_right or done_particles


# ============ SLICE DETECTION ============
def check_slice(obj, tracker, finger_pos):
    if obj.sliced or obj.slice_cooldown > 0:
        return False

    if finger_pos:
        dx = finger_pos[0] - obj.x
        dy = finger_pos[1] - obj.y
        if math.sqrt(dx**2 + dy**2) < obj.radius + 20:
            return True

    points = tracker.get_slice_points()
    for px, py in points:
        dx = px - obj.x
        dy = py - obj.y
        if math.sqrt(dx**2 + dy**2) < obj.radius + 18:
            return True

    if len(points) >= 2:
        for i in range(len(points) - 1):
            p1 = points[i]; p2 = points[i + 1]
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
            denom = dx*dx + dy*dy
            t = ((obj.x - p1[0])*dx + (obj.y - p1[1])*dy) / denom
            t = max(0.0, min(1.0, t))
            cx = p1[0] + t*dx; cy = p1[1] + t*dy
            if math.sqrt((obj.x-cx)**2 + (obj.y-cy)**2) < obj.radius + 15:
                return True

    if tracker.predicted_pos:
        px, py = tracker.predicted_pos
        dx = px - obj.x; dy = py - obj.y
        if math.sqrt(dx**2 + dy**2) < obj.radius + 15:
            return True

    return False


# ============ GAME STATE ============
fruits       = []
score        = 0
high_score   = 0
lives        = 3
spawn_timer  = 0
finger_pos   = None
trail        = deque(maxlen=30)
game_over    = False
combo        = 0
combo_timer  = 0
shake_timer  = 0
finger_tracker = FingerTracker(history_size=10)

high_score_file = os.path.join(os.path.dirname(__file__), 'highscore.txt')
try:
    with open(high_score_file, 'r') as f:
        high_score = int(f.read())
except:
    high_score = 0

print("=" * 50)
print("🍉 FRUIT NINJA - ENHANCED TRACKING 🍉")
print("=" * 50)
print("Peningkatan:")
print("- Buah lebih cepat & loncatan lebih tinggi")
print("- Buah bisa muncul dari kiri & kanan layar")
print("- Smoothing adaptif & prediksi posisi jari")
print("- 4 metode deteksi slice")
print("Tekan: 'q' keluar | 'r' restart")
print("=" * 50)

# ============ MAIN LOOP ============
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Screen shake
    if shake_timer > 0:
        shake_timer -= 1
        M = np.float32([[1, 0, random.randint(-8,8)], [0, 1, random.randint(-8,8)]])
        frame = cv2.warpAffine(frame, M, (w, h))

    # Background terang
    overlay = frame.copy()
    cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Deteksi tangan
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, int(time.time() * 1000))

    if latest_result and latest_result.hand_landmarks and not game_over:
        hand  = latest_result.hand_landmarks[0]
        tip   = hand[8]
        raw_x = int(tip.x * w)
        raw_y = int(tip.y * h)
        finger_tracker.update(raw_x, raw_y)
        sx, sy     = finger_tracker.smooth(raw_x, raw_y)
        finger_pos = (sx, sy)
        trail.append(finger_pos)
    else:
        finger_pos = None
        if trail:
            trail.popleft()

    # ============ SPAWN ============
    if not game_over:
        spawn_timer += 1
        delay = max(18, 45 - score // 2)   # Spawn lebih sering seiring skor naik

        if spawn_timer > delay:
            bomb_chance = min(0.35, 0.1 + score * 0.005)

            # Spawn utama
            x, y, vx, vy = spawn_fruit(w, h, score)
            if random.random() < bomb_chance:
                fruits.append(GameObject(x, h + 50, vx, vy, 'bomb'))
            else:
                fruits.append(GameObject(x, y, vx, vy, 'fruit', random.choice(fruit_names)))

            # 30% chance spawn buah kedua
            if random.random() < 0.30 and score > 2:
                x2, y2, vx2, vy2 = spawn_fruit(w, h, score)
                fruits.append(GameObject(x2, y2, vx2, vy2, 'fruit', random.choice(fruit_names)))

            spawn_timer = 0

    # ============ UPDATE & DRAW OBJECTS ============
    for obj in fruits[:]:
        obj.update()
        if obj.sliced:
            obj.update_particles()

        if not obj.sliced and not game_over:
            if check_slice(obj, finger_tracker, finger_pos):
                if obj.slice():
                    if obj.type == 'bomb':
                        game_over   = True
                        shake_timer = 20
                        cv2.putText(frame, "BOOM!", (w//2-60, h//2),
                                    cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,255), 5)
                    else:
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
                if lives <= 0:
                    game_over = True

    # Combo timer
    if combo_timer > 0:
        combo_timer -= 1
        if combo_timer == 0:
            combo = 0

    # ============ DRAW TRAIL ============
    if len(trail) >= 2:
        tlist = list(trail)
        for i in range(1, len(tlist)):
            a  = i / len(tlist)
            th = max(1, int(2 + a * 10))
            c  = (int(50*(1-a)), int(200*a), int(255*a))
            cv2.line(frame, tlist[i-1], tlist[i], c, th)

    # ============ DRAW KURSOR ============
    if finger_pos:
        oc = (0,100,255) if finger_tracker.speed > 400 else (100,100,255)
        ic = (0,200,255) if finger_tracker.speed > 400 else (150,200,255)
        cv2.circle(frame, finger_pos, 24, oc, 2)
        cv2.circle(frame, finger_pos, 18, ic, 2)
        cv2.circle(frame, finger_pos,  8, (255,255,255), -1)
        cv2.circle(frame, finger_pos,  4, (0,255,255), -1)
        if finger_tracker.speed > 200 and finger_tracker.predicted_pos:
            px = max(0, min(w-1, finger_tracker.predicted_pos[0]))
            py = max(0, min(h-1, finger_tracker.predicted_pos[1]))
            cv2.arrowedLine(frame, finger_pos, (px,py), (0,255,200), 2, tipLength=0.4)

    # ============ UI ============
    cv2.putText(frame, f"Score: {score}",   (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1,   (255,255,255), 2)
    cv2.putText(frame, f"Best:  {high_score}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    spd = min(int(finger_tracker.speed), 999)
    sc  = (0,255,0) if spd < 200 else (0,255,255) if spd < 400 else (0,100,255)
    cv2.putText(frame, f"Speed: {spd}", (15, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 1)

    for i in range(3):
        hx = w - 120 + i * 35
        lc = (0,0,255) if i >= lives else (80,80,80)
        lt = 3         if i >= lives else 2
        cv2.putText(frame, "X", (hx, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, lc, lt)

    if combo >= 3:
        cv2.putText(frame, f"COMBO x{combo}!", (w//2-80, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,255,255), 3)

    if game_over:
        go_ov = frame.copy()
        cv2.rectangle(go_ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(go_ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "GAME OVER",      (w//2-130, h//2-30), cv2.FONT_HERSHEY_DUPLEX,  2,   (0,0,255),     4)
        cv2.putText(frame, f"Score: {score}",(w//2-80,  h//2+25), cv2.FONT_HERSHEY_DUPLEX,  1.2, (255,255,255), 2)
        cv2.putText(frame, "Tekan 'r' main lagi", (w//2-120, h//2+70),  cv2.FONT_HERSHEY_SIMPLEX, 0.65,(200,200,200),2)
        cv2.putText(frame, "atau 'q' keluar",      (w//2-90,  h//2+105), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(150,150,150),2)
        if score > high_score:
            high_score = score
            try:
                with open(high_score_file, 'w') as f:
                    f.write(str(high_score))
            except:
                pass

    if not game_over:
        cv2.putText(frame, "'r' restart | 'q' quit", (w//2-90, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

    cv2.imshow('Fruit Ninja - Enhanced Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        fruits.clear()
        score = 0; lives = 3; combo = 0
        combo_timer = 0; spawn_timer = 0
        game_over = False; shake_timer = 0
        trail.clear()
        finger_tracker = FingerTracker(history_size=10)

# Cleanup
landmarker.close()
cap.release()
cv2.destroyAllWindows()
print(f"\nFinal Score: {score} | High Score: {high_score}")
print("Terima kasih sudah bermain!")