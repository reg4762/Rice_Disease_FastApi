import os
from dotenv import load_dotenv

# Muat file .env
load_dotenv()

# Path ke model
MODEL_PATH = os.getenv("MODEL_PATH", "models/paddy_disease_model.h5")
