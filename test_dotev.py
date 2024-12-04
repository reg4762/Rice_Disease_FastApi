from dotenv import load_dotenv
import os

# Memuat file .env
load_dotenv()

# Mengakses variabel lingkungan
api_key = os.getenv("API_KEY")
debug = os.getenv("DEBUG")

print(f"API_KEY: {api_key}")
print(f"DEBUG: {debug}")
