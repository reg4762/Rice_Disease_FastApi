from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.image_utils import preprocess_image
from app.models.model_loader import load_model
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse

router = APIRouter()
model = load_model()  

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Validasi tipe file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")

        
        image = Image.open(file.file)
        image = image.convert("RGB")  
        image = image.resize((224, 224))  
        image_array = np.array(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=0)  

        # Prediksi menggunakan model
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])  # Ambil kelas dengan probabilitas tertinggi
        confidence = float(np.max(predictions[0]))  # Ambil confidence level

        # Mapping hasil prediksi ke label
        labels = ["Healthy", "Bacterial Leaf Blight", "Brown Spot", "Leaf Scald", "Leaf Blast", "Narrow Brown Spot"]
        result = labels[predicted_class]

        return {"class": result, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {e}")
