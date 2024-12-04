from fastapi import Depends
from app.core.config import settings
from app.models.model_loader import load_model

# Dependency untuk memuat model
def get_model():
    return load_model(settings.MODEL_PATH)
