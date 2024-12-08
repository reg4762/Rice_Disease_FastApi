import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from io import BytesIO
from PIL import Image

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Memuat model yang sudah dilatih (pastikan model berada di lokasi yang benar)
model = load_model('models/rice_leaf_disease.h5')

# Compile model dengan optimizer dan metrik
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Fungsi untuk memproses gambar agar sesuai dengan input model
def preprocess_image(img):
    img = img.convert("RGB")  # Konversi gambar ke format RGB
    img = img.resize((128, 128))  # Sesuaikan ukuran gambar dengan model
    img_array = np.array(img)  # Konversi gambar ke array numpy
    img_array = img_array.astype('float32')  # Konversi array ke float32
    img_array /= 255.0  # Normalisasi ke [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    return img_array

# Fungsi untuk memprediksi penyakit berdasarkan gambar
def predict_disease(img, threshold=0.85):
    """
    Fungsi untuk memprediksi jenis penyakit dari gambar yang sudah diproses.
    - Mengembalikan kelas penyakit yang diprediksi dan probabilitasnya.
    - Mengembalikan 'unknown' jika probabilitas di bawah threshold.
    """
    img_array = preprocess_image(img)  # Preprocess gambar
    prediction = model.predict(img_array)  # Melakukan prediksi dengan model
    class_labels = sorted(os.listdir('dataset/train'))  # Daftar kelas penyakit
    predicted_class_idx = np.argmax(prediction, axis=1)[0]  # Mendapatkan indeks kelas tertinggi
    probability = float(prediction[0][predicted_class_idx])  # Probabilitas kelas tertinggi

    if probability < threshold:  # Jika probabilitas di bawah threshold
        return "unknown", probability

    predicted_disease = class_labels[predicted_class_idx]  # Nama kelas penyakit
    return predicted_disease, probability

# Endpoint untuk menerima gambar dan memberikan prediksi
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint untuk menerima gambar yang diupload dan memberikan prediksi jenis penyakit.
    """
    try:
        image_data = await file.read()  # Membaca data gambar yang diupload
        img = Image.open(BytesIO(image_data))  # Membuka gambar dengan PIL

        # Prediksi penyakit berdasarkan gambar yang diupload
        predicted_disease, probability = predict_disease(img)

        # Mengembalikan hasil prediksi dalam bentuk JSON
        return {
            "predicted_disease": predicted_disease,
            "probability": probability
        }
    except Exception as e:
        return {"error": str(e)}

