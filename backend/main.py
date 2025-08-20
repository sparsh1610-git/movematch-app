from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
import os
from move_matcher import auto_compare  # your function

app = FastAPI()

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_files(test_video: UploadFile = File(...), ref_video: UploadFile = File(...)):
    # Save uploaded files
    test_path = os.path.join(UPLOAD_FOLDER, "test.mp4")
    ref_path = os.path.join(UPLOAD_FOLDER, "ref.mp4")
    
    with open(test_path, "wb") as buffer:
        shutil.copyfileobj(test_video.file, buffer)
    with open(ref_path, "wb") as buffer:
        shutil.copyfileobj(ref_video.file, buffer)

    # Here: run your pose extraction and comparison
    score = auto_compare(test_path, ref_path)

    return {"score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
