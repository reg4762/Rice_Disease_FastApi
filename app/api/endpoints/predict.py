# from fastapi import APIRouter, File, UploadFile, HTTPException
# from app.utils.image_utils import preprocess_image
# from app.models.model_loader import load_model
# import numpy as np
# from PIL import Image
# from fastapi.responses import JSONResponse

# router = APIRouter()
# model = load_model()  # Pastikan model sesuai dengan pipeline pelatihan

# # Label sesuai urutan pelatihan model
# labels = ["Healthy", "Bacterial Leaf Blight", "Brown Spot", "Leaf Scald", "Leaf Blast", "Narrow Brown Spot"]

# @router.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Validasi tipe file
#         if not file.content_type.startswith("image/"):
#             raise HTTPException(status_code=400, detail="File harus berupa gambar")

#         # Buka file dan preprocessing gambar
#         image = Image.open(file.file)
#         image = image.convert("RGB")  # Pastikan format gambar RGB
#         image = image.resize((128, 128))  
#         image_array = np.array(image) / 255.0  # Normalisasi ke [0,1]
#         image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch

#         # Prediksi menggunakan model
#         predictions = model.predict(image_array)
#         predicted_class = np.argmax(predictions[0])  # Ambil kelas dengan probabilitas tertinggi
#         confidence = float(np.max(predictions[0]))  # Ambil confidence level

#         # Mapping hasil prediksi ke label
#         result = labels[predicted_class]

#         return {"class": result, "confidence": confidence}

#     except Exception as e:
#         # Debugging kesalahan jika ada
#         raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
