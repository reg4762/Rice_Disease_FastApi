# from PIL import Image
# import numpy as np

# def preprocess_image(image_bytes, target_size=(224, 224)):
#     image = Image.open(image_bytes).convert("RGB")
#     image = image.resize(target_size)
#     image_array = np.array(image) / 255.0  # Normalisasi ke [0, 1]
#     return np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension
