import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Setup window
root = tk.Tk()
root.title("Camera Test")
label = tk.Label(root)
label.pack()

# Buka kamera
cap = cv2.VideoCapture(0)

def show_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(30, show_frame)

show_frame()
root.mainloop()
cap.release()