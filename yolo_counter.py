import cv2, time, threading
from datetime import datetime, timezone
from collections import deque
import numpy as np
from ultralytics import YOLO
from pymongo.collection import Collection
from sms_alert import send_sms_alert  # ✅ Add this import


class CrowdMonitor:
    """
    Runs YOLO detection in a background thread for ONE location.
    - Writes counts to MongoDB 'readings'.
    - Holds latest JPEG frame for MJPEG streaming.
    - Start/Stop + live capacity update.
    """
    def __init__(self, location_name, source, capacity, readings_col: Collection,
                 roi_points=None, confidence=0.3, smooth_window=10, alert_hold_seconds=3):
        self.location_name = location_name
        self.source = 0 if str(source) == "0" else str(source)
        self.capacity = int(capacity)
        self.readings_col = readings_col
        self.roi_points = roi_points
        self.confidence = float(confidence)
        self.smooth_window = int(smooth_window)
        self.alert_hold_seconds = int(alert_hold_seconds)

        self._model = YOLO("yolov8n.pt")
        self._cap = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._counts = deque(maxlen=self.smooth_window)
        self._exceed_start = None
        self._alert_active = False
        self._latest_jpeg = None
        self.running = False

        self._alert_sent = False  # ✅ Flag to prevent repeated SMS

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.running = True

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None
        self.running = False

    def update_capacity(self, new_capacity: int):
        with self._lock:
            self.capacity = int(new_capacity)

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def _inside_roi(self, cx, cy):
        if not self.roi_points:
            return True
        cnt = np.array(self.roi_points, dtype=np.int32)
        return cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False) >= 0

    def _encode(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buf.tobytes() if ok else None

    def _run(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            print(f"[{self.location_name}] Could not open source: {self.source}")
            self.running = False
            return

        # Optional: speed up
        # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.2)
                continue

            # Detect people
            res = self._model(frame, conf=self.confidence, classes=[0], verbose=False)
            boxes = []
            if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                boxes = res[0].boxes.xyxy.cpu().numpy().astype(int).tolist()

            # Count inside ROI
            count = 0
            for (x1, y1, x2, y2) in boxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if self._inside_roi(cx, cy):
                    count += 1

            self._counts.append(count)
            smooth = int(round(sum(self._counts) / max(1, len(self._counts))))

            # Alert hold logic
            now = time.time()
            if smooth > self.capacity:
                if self._exceed_start is None:
                    self._exceed_start = now
                elif (now - self._exceed_start) >= self.alert_hold_seconds:
                    self._alert_active = True
                    if not self._alert_sent:
                        print(f"[ALERT] {self.location_name}: {smooth} > {self.capacity}")
                        send_sms_alert(self.location_name, smooth, self.capacity)  # ✅ Send SMS
                        self._alert_sent = True  # ✅ Prevent resending until reset
            else:
                self._exceed_start = None
                self._alert_active = False
                self._alert_sent = False  # ✅ Reset SMS flag when safe

            # Write reading to DB
            try:
                self.readings_col.insert_one({
                    "location": self.location_name,
                    "ts": datetime.now(timezone.utc),
                    "count": smooth,
                    "alert": bool(self._alert_active)
                })
            except Exception as e:
                print("Mongo insert error:", e)

            # Draw overlay
            if self.roi_points:
                pts = np.array(self.roi_points, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            color = (0, 255, 0) if smooth <= self.capacity else (0, 0, 255)
            cv2.putText(frame, self.location_name, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(frame, f"People: {smooth}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Capacity: {self.capacity}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            if self._alert_active:
                cv2.putText(frame, "ALERT: Capacity exceeded", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

            # Store JPEG
            jpeg = self._encode(frame)
            with self._lock:
                self._latest_jpeg = jpeg

            time.sleep(0.05)

        if self._cap:
            self._cap.release()
            self._cap = None
