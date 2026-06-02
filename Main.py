import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Path ke file model
model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)
print("Kamera menyala! Tekan 'q' untuk keluar")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = landmarker.detect(mp_image)
        
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                h, w, _ = frame.shape
                
                # Gambar titik-titik
                for lm in hand:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), cv2.FILLED)
                
                # Ujung jari telunjuk
                tip = hand[8]
                tx, ty = int(tip.x * w), int(tip.y * h)
                cv2.circle(frame, (tx, ty), 15, (0, 0, 255), cv2.FILLED)
        
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Selesai!")