# import numpy as np


# def format_response(predictions):
#     classes = ["Healthy", "Bacterial Blight", "Blast", "Brown Spot"]
#     probabilities = {label: float(prob) for label, prob in zip(classes, predictions[0])}
#     return {
#         "prediction": classes[np.argmax(predictions)],
#         "probabilities": probabilities
#     }
