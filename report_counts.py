# report_counts.py
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "crowd_counter_db")

if not MONGODB_URI:
    raise SystemExit("Please set MONGODB_URI in your .env file.")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
locations = db.locations
readings = db.readings

def pick_location_name():
    locs = list(locations.find({}, {"_id": 0, "name": 1}).sort("name", 1))
    if not locs:
        raise SystemExit("No locations in DB.")
    print("\nSelect location:")
    for i, d in enumerate(locs, start=1):
        print(f"[{i}] {d['name']}")
    idx = input("Enter number: ").strip()
    try:
        idx = int(idx) - 1
        assert 0 <= idx < len(locs)
    except:
        raise SystemExit("Invalid selection.")
    return locs[idx]["name"]

def main():
    name = pick_location_name()
    hours_str = input("How many past hours to include? (default 2): ").strip() or "2"
    try:
        hours = int(hours_str)
    except:
        hours = 2

    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(hours=hours)

    # Query readings
    cur = readings.find(
        {"location": name, "ts": {"$gte": start_utc, "$lte": end_utc}},
        {"_id": 0, "ts": 1, "count": 1, "alert": 1}
    ).sort("ts", 1)

    rows = list(cur)
    if not rows:
        print("No data in that time range.")
        return

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")

    # Resample per minute (mean count)
    per_min = df["count"].resample("1min").mean().fillna(0)

    # Save CSV
    out_csv = f"{name}_last_{hours}h.csv"
    per_min.to_csv(out_csv, header=["avg_count"])
    print(f"Saved CSV: {out_csv}")

    # Plot
    plt.figure()
    per_min.plot(title=f"{name} — average count per minute (last {hours}h)")
    plt.xlabel("Time")
    plt.ylabel("People (avg/min)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
