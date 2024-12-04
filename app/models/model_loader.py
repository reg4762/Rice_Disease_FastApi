import tensorflow as tf
import os

def load_model():
    try:
        path_model = os.getenv("MODEL_PATH", "models/paddy_disease_model.h5")
        model = tf.keras.models.load_model(path_model)
        print("Model berhasil dimuat.")
        return model
    except Exception as e:
        raise RuntimeError(f"Error saat memuat model: {e}")
    

def predict_disease(model, image):
    # Predict the disease
    predictions = model.predict(image)
    class_index = predictions.argmax(axis=-1)[0]
    class_names = ["Healthy", "Brown Spot", "Blast", "Sheath Blight"]  # Example classes
    return class_names[class_index]
    
