import os
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai

# Load .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "multimodal_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chunks")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing in .env")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Mongo client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]
chunks_collection = db[MONGODB_COLLECTION]
