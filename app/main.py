from fastapi import FastAPI
from app.api.endpoints.predict import router as predict_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Rice Disease Detection API"}

# Include the predict router
app.include_router(predict_router, prefix="/detect", tags=["Detect"])

