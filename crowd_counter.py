import os
import time
from collections import deque
from datetime import datetime, timezone


import cv2
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
from dotenv import load_dotenv

# Optional sound
ALERT_SOUND_FILE = "alert.wav"  # put a sound file here (wav or mp3). Leave as None to disable.
USE_SOUND = True

# -----------------------------
# Load .env config
# -----------------------------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "crowd_counter_db")
LOCATION_NAME = os.getenv("DEFAULT_LOCATION", "Demo-Bridge")

# Video source (0 = webcam). For real camera later, replace with RTSP URL string.
SOURCE = 0

# Detection / alert settings (you can tune later)
CONFIDENCE = 0.3
SMOOTH_WINDOW = 10
ALERT_HOLD_SECONDS = 3
SHOW_BOXES = True
ROI_POINTS = None  # Example: [(100,200),(1200,220),(1180,700),(130,720)]

# -----------------------------
# MongoDB connect
# -----------------------------
if not MONGODB_URI:
    raise SystemExit("Please set MONGODB_URI in your .env file.")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
locations_col = db.locations
readings_col = db.readings

# Ensure indexes (helps queries later; safe to run many times)
locations_col.create_index("name", unique=True)
readings_col.create_index([("location", 1), ("ts", -1)])

# Get capacity from DB
loc_doc = locations_col.find_one({"name": LOCATION_NAME})
if not loc_doc:
    raise SystemExit(f"Location '{LOCATION_NAME}' not found. Run seed_locations.py first.")
CAPACITY = int(loc_doc.get("capacity", 200))

print(f"Using location: {LOCATION_NAME} (capacity = {CAPACITY})")

# -----------------------------
# Helper functions
# -----------------------------
def inside_roi(cx, cy, roi_pts):
    if not roi_pts:
        return True
    cnt = np.array(roi_pts, dtype=np.int32)
    return cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False) >= 0

def draw_roi(frame, roi_pts):
    if not roi_pts:
        return
    pts = np.array(roi_pts, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

def put_text(frame, text, org, color=(255, 255, 255), scale=0.9, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def play_alert_sound():
    if not USE_SOUND or not ALERT_SOUND_FILE:
        return
    try:
        from playsound import playsound
        playsound(ALERT_SOUND_FILE, block=False)
    except Exception as e:
        print("(Sound) Could not play alert sound:", e)

# -----------------------------
# Load model and open camera
# -----------------------------
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print("Error loading YOLO model:", e)
    raise SystemExit(1)

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("Could not open video source. If you used 0, try another index like 1.")
    raise SystemExit(1)

# Reduce resolution if your laptop is slow (optional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

counts = deque(maxlen=SMOOTH_WINDOW)
exceed_start = None
alert_active = False
fps_prev_t = time.time()

print("Press 'q' to quit. Press 'b' to toggle boxes on/off.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Frame not received. Reconnecting...")
        time.sleep(0.5)
        continue

    # Detect persons
    results = model(frame, conf=CONFIDENCE, classes=[0], verbose=False)
    boxes = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            boxes = xyxy.tolist()

    # Count inside ROI (or whole frame)
    count = 0
    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if inside_roi(cx, cy, ROI_POINTS):
            count += 1

    counts.append(count)
    smooth_count = int(round(sum(counts) / max(1, len(counts))))

    # Alert logic with hold time
    now = time.time()
    if smooth_count > CAPACITY:
        if exceed_start is None:
            exceed_start = now
        elif (now - exceed_start) >= ALERT_HOLD_SECONDS and not alert_active:
            alert_active = True
            print(f"[ALERT] {LOCATION_NAME}: {smooth_count} > {CAPACITY}")
            play_alert_sound()
    else:
        exceed_start = None
        alert_active = False

    # Log to MongoDB every second (approx)
    # You can change the frequency later if you want less data.
    ts = datetime.now(timezone.utc)

    try:
        readings_col.insert_one({
            "location": LOCATION_NAME,
            "ts": ts,
            "count": smooth_count,
            "alert": bool(alert_active)
        })
    except Exception as e:
        print("MongoDB insert error:", e)

    # Draw overlays
    draw_roi(frame, ROI_POINTS)
    status_color = (0, 255, 0) if smooth_count <= CAPACITY else (0, 0, 255)
    put_text(frame, f"People: {smooth_count}", (20, 40), color=status_color, scale=1.0)
    put_text(frame, f"Capacity: {CAPACITY}", (20, 75))
    if alert_active:
        put_text(frame, "ALERT: Capacity exceeded", (20, 110), color=(0, 0, 255), scale=1.0, thickness=3)

    if SHOW_BOXES:
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if inside_roi(cx, cy, ROI_POINTS):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

    # FPS display
    t = time.time()
    fps = 1.0 / max(1e-6, (t - fps_prev_t))
    fps_prev_t = t
    put_text(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20))

    cv2.imshow("Crowd Counter (with DB)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        SHOW_BOXES = not SHOW_BOXES

cap.release()
cv2.destroyAllWindows()
