import os

# Struktur direktori yang diinginkan
project_structure = {
        'app': {
            '__init__.py': '',
            'main.py': '''from fastapi import FastAPI
from app.api.endpoints import predict

app = FastAPI(title="Paddy Disease Detection API")

# Tambahkan router endpoint
app.include_router(predict.router, prefix="/api/v1")
''',
            'api': {
                '__init__.py': '',
                'endpoints': {
                    '__init__.py': '',
                    'predict.py': '''from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.model_loader import load_model
from app.utils.image_utils import preprocess_image
from app.utils.response_utils import format_response

router = APIRouter()

# Memuat model hanya sekali saat server berjalan
model = load_model()

@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Baca dan preprocess gambar
        img = await image.read()
        preprocessed_img = preprocess_image(img)

        # Prediksi penyakit menggunakan CNN
        prediction = model.predict(preprocessed_img)

        # Format hasil prediksi
        response = format_response(prediction)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
''',
                },
                'dependencies.py': '''from fastapi import Depends
from app.core.config import settings
from app.models.model_loader import load_model

# Dependency untuk memuat model
def get_model():
    return load_model(settings.MODEL_PATH)
'''
            },
            'core': {
                '__init__.py': '',
                'config.py': '''import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/paddy_disease_model.h5")

settings = Settings()
'''
            },
            'models': {
                '__init__.py': '',
                'cnn_model.py': '''from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    return model
''',
                'model_loader.py': '''from tensorflow.keras.models import load_model

def load_model(model_path="models/paddy_disease_model.h5"):
    return load_model(model_path)
'''
            },
            'utils': {
                '__init__.py': '',
                'image_utils.py': '''from PIL import Image
import numpy as np

def preprocess_image(image_bytes, target_size=(224, 224)):
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalisasi ke [0, 1]
    return np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension
''',
                'response_utils.py': '''def format_response(predictions):
    labels = ["Healthy", "Bacterial Blight", "Blast", "Brown Spot"]
    probabilities = {label: float(prob) for label, prob in zip(labels, predictions[0])}
    return {
        "prediction": labels[np.argmax(predictions)],
        "probabilities": probabilities
    }
'''
            }
        },
        'models': {
            'paddy_disease_model.h5': ''  # Placeholder untuk model CNN
        },
        'static': {
            'example.jpg': ''  # Placeholder untuk contoh gambar
        },
        'requirements.txt': '''fastapi
uvicorn
tensorflow
pillow
numpy
python-dotenv
''',
        '.env': '''MODEL_PATH=models/paddy_disease_model.h5
''',
        'README.md': '''# Paddy Disease Detection API
API untuk mendeteksi penyakit pada daun padi menggunakan model CNN.
'''
    }


# Fungsi untuk membuat struktur direktori dan file
def create_structure(base_path, structure):
    for name, value in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(value, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, value)
        else:
            with open(path, 'w') as f:
                f.write(value)

if __name__ == '__main__':
    # Tentukan nama folder proyek
    project_name = 'fastapi-paddy-disease'

    # Buat struktur proyek
    create_structure(os.getcwd(), project_structure)

    print(f"Proyek '{project_name}' berhasil dibuat!")
