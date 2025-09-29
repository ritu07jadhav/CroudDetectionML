# app.py
import os
from datetime import datetime, timedelta, timezone
from threading import Lock

from flask import (
    Flask, render_template, Response, request, redirect,
    url_for, send_from_directory, flash
)
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd

from yolo_counter import CrowdMonitor

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "crowd_counter_db")
if not MONGODB_URI:
    raise SystemExit("Set MONGODB_URI in .env (MONGODB_URI=...)")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
LOC = db.locations
READ = db.readings
REPO = db.reports  # metadata of generated CSVs
ALERT_WAV_PATH = os.path.join(os.getcwd(), "alert.wav")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")

REPORT_DIR = os.path.join(os.getcwd(), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Single worker (single camera UI)
workers_lock = Lock()
worker = None
worker_name = None

def _to_source(v):
    return 0 if str(v) == "0" else str(v)

def get_current_location():
    """Return the first location (or create a default)."""
    doc = LOC.find_one({}, sort=[("name", 1)])
    if not doc:
        default = {"name": "Demo-Bridge", "capacity": 200, "source": "0"}
        LOC.insert_one(default)
        return default
    return {k: doc[k] for k in doc if k != "_id"}

@app.route("/")
def home():
    return redirect(url_for("live"))

# ------------- LIVE -------------
@app.route("/live")
def live():
    loc = get_current_location()
    is_running = False
    with workers_lock:
        global worker, worker_name
        if worker and worker_name == loc["name"] and worker.running:
            is_running = True
    return render_template("live.html", loc=loc, running=is_running, active="live")

@app.route("/start", methods=["POST"])
def start_camera():
    loc = get_current_location()
    with workers_lock:
        global worker, worker_name
        if worker and worker_name == loc["name"] and worker.running:
            flash("Camera already running.")
        else:
            mon = CrowdMonitor(
                location_name=loc["name"],
                source=_to_source(loc.get("source", "0")),
                capacity=int(loc.get("capacity", 200)),
                readings_col=READ,
                roi_points=loc.get("roi")
            )
            mon.start()
            worker = mon
            worker_name = loc["name"]
            flash(f"Started camera for {worker_name}.")
    return redirect(url_for("live"))

@app.route("/stop", methods=["POST"])
def stop_camera():
    with workers_lock:
        global worker, worker_name
        if worker and worker.running:
            worker.stop()
            flash(f"Stopped camera for {worker_name}.")
        else:
            flash("Camera not running.")
        worker = None
        worker_name = None
    return redirect(url_for("live"))

@app.route("/stream")
def stream():
    def gen():
        while True:
            with workers_lock:
                mon = worker
            if not mon or not mon.running:
                import time; time.sleep(0.2); continue
            frame = mon.get_latest_jpeg()
            if not frame:
                import time; time.sleep(0.03); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------- SETTINGS -------------
@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        # stop camera on save so changes apply cleanly
        with workers_lock:
            global worker, worker_name
            if worker and worker.running:
                worker.stop()
            worker = None
            worker_name = None

        # Read the current (old) location from DB so we can update by old name
        current_loc = get_current_location()
        old_name = current_loc.get("name", "Demo-Bridge")

        new_name = (request.form.get("name") or "").strip() or "Demo-Bridge"
        cap_str  = (request.form.get("capacity") or "200").strip()
        source   = (request.form.get("source") or "0").strip()

        try:
            capacity = int(cap_str)
        except Exception:
            flash("Capacity must be a whole number.")
            return redirect(url_for("settings"))

        # Update existing location by *old* name; insert if nothing matched
        res = LOC.update_one(
            {"name": old_name},
            {"$set": {"name": new_name, "capacity": capacity, "source": source}},
            upsert=False
        )
        if res.matched_count == 0:
            # First time or doc missing — insert a fresh one
            LOC.insert_one({"name": new_name, "capacity": capacity, "source": source})

        # If name changed, migrate historical data to keep reports consistent
        if new_name != old_name:
            READ.update_many({"location": old_name}, {"$set": {"location": new_name}})
            REPO.update_many({"location": old_name}, {"$set": {"location": new_name}})

        flash("Settings saved. Start the camera from the Live page.")
        return redirect(url_for("live"))

    loc = get_current_location()
    return render_template("settings.html", loc=loc, active="settings")

# ------------- REPORTS -------------
@app.route("/reports", methods=["GET", "POST"])
def reports():
    loc = get_current_location()
    if request.method == "POST":
        hours_str = request.form.get("hours", "2")
        try:
            hours = int(hours_str)
        except:
            hours = 2

        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(hours=hours)

        rows = list(READ.find(
            {"location": loc["name"], "ts": {"$gte": start_utc, "$lte": end_utc}},
            {"_id": 0, "ts": 1, "count": 1, "alert": 1}
        ).sort("ts", 1))

        if not rows:
            flash("No data in that time range.")
            return redirect(url_for("reports"))

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).astype("datetime64[ns, UTC]")
        df = df.set_index("ts")
        per_min = df["count"].resample("1min").mean().fillna(0)

        ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{loc['name']}_last_{hours}h_{ts_label}.csv"
        out_path = os.path.join(REPORT_DIR, filename)
        per_min.to_csv(out_path, header=["avg_count"])

        REPO.insert_one({
            "location": loc["name"],
            "created_at": datetime.now(timezone.utc),
            "range_hours": hours,
            "file": filename,
            "rows": int(per_min.shape[0])
        })

        flash(f"Report created: {filename}")
        return redirect(url_for("reports"))

    # GET: list reports (latest first)
    report_rows = list(REPO.find({"location": loc["name"]}).sort("created_at", -1))
    return render_template("reports.html", loc=loc, reports=report_rows, active="reports")

@app.route("/reports/download/<path:filename>")
def download_report(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

# ------------ ALERTS (sound + status) ------------
@app.route("/alert.wav")
def alert_wav():
    return send_from_directory(
        os.path.dirname(ALERT_WAV_PATH),
        os.path.basename(ALERT_WAV_PATH),
        mimetype="audio/wav",
        as_attachment=False,
        max_age=0
    )

@app.route("/alert_status")
def alert_status():
    loc = get_current_location()
    row = READ.find_one(
        {"location": loc["name"]},
        sort=[("ts", -1)],
        projection={"_id": 0, "alert": 1}
    )
    return {"alert": bool(row and row.get("alert"))}

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    # No reloader → avoids Windows socket error
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
