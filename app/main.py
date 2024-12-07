import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam  # Import Adam
from io import BytesIO
from PIL import Image

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Memuat model yang sudah dilatih (pastikan model berada di lokasi yang benar)
model = load_model('models/rice_leaf_disease.h5')

# Compile model dengan optimizer dan metrik
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# Fungsi untuk memproses gambar agar sesuai dengan input model
def preprocess_image(img_path):
    # Jika img_path sudah berupa objek gambar, langsung gunakan objek tersebut
    if isinstance(img_path, Image.Image):
        img = img_path
    else:
        img = Image.open(img_path)  # Membuka gambar jika path diberikan

    img = img.resize((128, 128))  # Sesuaikan ukuran gambar dengan model
    img_array = np.array(img)  # Konversi gambar ke array numpy

    # Konversi array gambar ke tipe data float32
    img_array = img_array.astype('float32')

    # Normalisasi gambar menjadi [0, 1]
    img_array /= 255.0

    # Perlu memastikan bentuk input sesuai, misalnya (1, 128, 128, 3) untuk gambar RGB
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
    
    return img_array


# Fungsi untuk memprediksi penyakit berdasarkan gambar
# Fungsi untuk memprediksi penyakit berdasarkan gambar
def predict_disease(img_path):
    """
    Fungsi untuk memprediksi jenis penyakit dari gambar yang sudah diproses.
    - Melakukan prediksi dengan model yang sudah dilatih.
    - Mengembalikan kelas penyakit yang diprediksi dan probabilitasnya.
    """
    img = Image.open(img_path)  # Membuka gambar dengan PIL
    img = preprocess_image(img)  # Preprocess gambar untuk model

    # Pastikan gambar memiliki dimensi yang benar (128, 128, 3)
    img = np.array(img)  # Mengonversi ke numpy array
    img = np.resize(img, (128, 128, 3))  # Menyesuaikan ukuran jika diperlukan
    
    # Memastikan gambar memiliki dimensi (128, 128, 3) dan menambah dimensi batch
    img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch, menjadi (1, 128, 128, 3)

    # Melakukan prediksi dengan model
    prediction = model.predict(img)  # Pastikan gambar sesuai input model

    # Mengambil nama kelas berdasarkan prediksi
    class_labels = sorted(os.listdir('dataset/train'))  # Pastikan path ke direktori kelas benar
    predicted_class_idx = np.argmax(prediction, axis=1)[0]  # Mendapatkan indeks kelas dengan nilai tertinggi
    predicted_disease = class_labels[predicted_class_idx]  # Nama kelas penyakit yang diprediksi
    probability = float(prediction[0][predicted_class_idx]) * 100  # Mengonversi ke float

    return predicted_disease, probability


# Endpoint untuk menerima gambar dan memberikan prediksi
@app.post("/predict/")  # Endpoint untuk menerima gambar dan memberikan prediksi
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()  # Membaca data gambar yang diupload

    # Mengonversi data gambar menjadi gambar PIL
    img = Image.open(BytesIO(image_data))

    # Menyimpan gambar sementara untuk diproses
    img_path = "temp_image.jpg"
    img.save(img_path)

    # Prediksi penyakit berdasarkan gambar yang diupload
    predicted_disease, probability = predict_disease(img_path)

    # Mengembalikan hasil prediksi dalam bentuk JSON
    return {
        "predicted_disease": predicted_disease,
        "probability": probability
    }

