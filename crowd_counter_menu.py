# crowd_counter_menu.py
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
from dotenv import load_dotenv

# Optional sound settings
USE_SOUND = True
ALERT_SOUND_FILE = "alert.wav"  # place a small wav/mp3 in project folder, or set to None

# Detection / alert defaults (can tune)
CONFIDENCE = 0.3
SMOOTH_WINDOW = 10
ALERT_HOLD_SECONDS = 3

def play_alert_sound():
    if not USE_SOUND or not ALERT_SOUND_FILE:
        return
    try:
        from playsound import playsound
        # If playsound on your OS blocks, remove block=False
        playsound(ALERT_SOUND_FILE, block=False)
    except Exception as e:
        print("(Sound) Could not play alert sound:", e)

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

def pick_location(locations_col):
    locs = list(locations_col.find({}, {"_id": 0}).sort("name", 1))
    if not locs:
        raise SystemExit("No locations in DB. Run manage_locations.py to add one.")
    print("\nSelect location:")
    for i, d in enumerate(locs, start=1):
        print(f"[{i}] {d.get('name')} (capacity={d.get('capacity')}, source={d.get('source','0')})")
    idx = input("Enter number: ").strip()
    try:
        idx = int(idx) - 1
        assert 0 <= idx < len(locs)
    except:
        raise SystemExit("Invalid selection.")
    return locs[idx]

def to_video_source(source_value):
    """
    Locations store 'source' as string:
    - "0" means use webcam index 0
    - Otherwise assume it's a full RTSP/HTTP URL string
    """
    if str(source_value) == "0":
        return 0
    return str(source_value)

def main():
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    DB_NAME = os.getenv("DB_NAME", "crowd_counter_db")

    if not MONGODB_URI:
        raise SystemExit("Please set MONGODB_URI in your .env file.")

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    locations_col = db.locations
    readings_col = db.readings
    locations_col.create_index("name", unique=True)
    readings_col.create_index([("location", 1), ("ts", -1)])

    # Pick which location to monitor
    loc = pick_location(locations_col)
    location_name = loc["name"]
    capacity = int(loc.get("capacity", 200))
    roi = loc.get("roi")  # optional: list of [x,y]
    source = to_video_source(loc.get("source", "0"))

    print(f"\nMonitoring: {location_name} | capacity={capacity} | source={source}\n")

    # Load model
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print("Error loading YOLO model:", e)
        return

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open video source. If webcam, try source index 1.")
        return

    # Optional downscale for speed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    counts = deque(maxlen=SMOOTH_WINDOW)
    exceed_start = None
    alert_active = False
    fps_prev_t = time.time()
    show_boxes = True  # <-- local flag replacing global SHOW_BOXES

    print("Press 'q' to quit. Press 'b' to toggle boxes on/off.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame not received...")
            time.sleep(0.3)
            continue

        results = model(frame, conf=CONFIDENCE, classes=[0], verbose=False)
        boxes = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                boxes = xyxy.tolist()

        # Count
        count = 0
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if inside_roi(cx, cy, roi):
                count += 1

        counts.append(count)
        smooth_count = int(round(sum(counts) / max(1, len(counts))))

        # Alert logic
        now = time.time()
        if smooth_count > capacity:
            if exceed_start is None:
                exceed_start = now
            elif (now - exceed_start) >= ALERT_HOLD_SECONDS and not alert_active:
                alert_active = True
                print(f"[ALERT] {location_name}: {smooth_count} > {capacity}")
                play_alert_sound()
        else:
            exceed_start = None
            alert_active = False

        # Log to DB
        ts = datetime.utcnow()
        try:
            readings_col.insert_one({
                "location": location_name,
                "ts": ts,
                "count": smooth_count,
                "alert": bool(alert_active)
            })
        except Exception as e:
            print("MongoDB insert error:", e)

        # Draw
        draw_roi(frame, roi)
        status_color = (0, 255, 0) if smooth_count <= capacity else (0, 0, 255)
        put_text(frame, f"{location_name}", (20, 35), scale=1.0)
        put_text(frame, f"People: {smooth_count}", (20, 70), color=status_color, scale=1.0)
        put_text(frame, f"Capacity: {capacity}", (20, 105))
        if alert_active:
            put_text(frame, "ALERT: Capacity exceeded", (20, 140), color=(0, 0, 255), scale=1.0, thickness=3)

        if show_boxes:
            for (x1, y1, x2, y2) in boxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if inside_roi(cx, cy, roi):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

        # FPS
        t = time.time()
        fps = 1.0 / max(1e-6, (t - fps_prev_t))
        fps_prev_t = t
        put_text(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20))

        cv2.imshow("Crowd Counter (menu)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            show_boxes = not show_boxes  # toggle local flag

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
