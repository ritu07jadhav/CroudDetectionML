# seed_locations.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "crowd_counter_db")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Demo-Bridge")
DEFAULT_CAPACITY = int(os.getenv("DEFAULT_CAPACITY", "200"))

if not MONGODB_URI:
    raise SystemExit("Please set MONGODB_URI in your .env file.")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

# Collection for locations (capacity per place)
locations = db.locations

# Upsert (insert if not exists, else update)
result = locations.update_one(
    {"name": DEFAULT_LOCATION},
    {"$set": {"name": DEFAULT_LOCATION, "capacity": DEFAULT_CAPACITY}},
    upsert=True
)

doc = locations.find_one({"name": DEFAULT_LOCATION})
print("Location is ready:")
print({"_id": str(doc.get("_id")), "name": doc.get("name"), "capacity": doc.get("capacity")})
