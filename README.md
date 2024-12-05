# Paddy Disease Detection API
API untuk mendeteksi penyakit pada daun padi menggunakan model CNN.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Technologies Used](#technologies-used)
                    
## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sandradevia/Rice_Disease_FastApi.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd Rice_Disease_FastAPi
   ```
3. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. **Install the Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the API:**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

2. **Access the Documentation:**
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Endpoints

- `POST /predict`: Upload an image file for prediction.
  - **Request:** Form-data with image file.
  - **Response:** JSON object with prediction results and accuracy.


## Technologies Used
- **FastAPI:** Web framework for building APIs.
- **TensorFlow/Keras:** Deep learning framework for building the detection model.
- **Python:** Programming language.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.
                      