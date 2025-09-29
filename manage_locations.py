# manage_locations.py
import os
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
locations.create_index("name", unique=True)

def list_locations():
    print("\n-- Locations --")
    for doc in locations.find({}, {"_id": 0}).sort("name", 1):
        print(f"- {doc.get('name')} | capacity={doc.get('capacity')} | source={doc.get('source', 0)}")
    print()

def add_or_update():
    name = input("Enter location name (e.g., Bridge-A): ").strip()
    if not name:
        print("Name is required.")
        return
    cap_str = input("Capacity (number, e.g., 200): ").strip() or "200"
    try:
        capacity = int(cap_str)
    except:
        print("Capacity must be a number.")
        return
    # Video source: keep 0 for laptop webcam for now; later put RTSP URL.
    source = input('Video source [0 for webcam or RTSP URL later] (default 0): ').strip() or "0"

    doc = {
        "name": name,
        "capacity": capacity,
        # store source as string; "0" means webcam index 0
        "source": source,
        # optional ROI polygon stored as list of [x,y] pairs
        # You can set later with roi_picker.py (optional step)
        # "roi": [[100,200],[1200,220],[1180,700],[130,720]],
    }
    locations.update_one({"name": name}, {"$set": doc}, upsert=True)
    print(f"Saved: {name} (capacity={capacity}, source={source})")

def delete_location():
    name = input("Enter location name to delete: ").strip()
    if not name:
        print("Name is required.")
        return
    res = locations.delete_one({"name": name})
    if res.deleted_count:
        print("Deleted.")
    else:
        print("Not found.")

def main():
    while True:
        print("""
[1] List locations
[2] Add or update a location
[3] Delete a location
[0] Quit
""")
        choice = input("Choose: ").strip()
        if choice == "1":
            list_locations()
        elif choice == "2":
            add_or_update()
        elif choice == "3":
            delete_location()
        elif choice == "0":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
