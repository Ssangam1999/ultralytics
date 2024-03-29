import os
import shutil
from fastapi import FastAPI, UploadFile, Query,File
import uvicorn
from enum import Enum
from typing import Optional
from ultralytics.models.yolo.segment.predict import SegmentationPredictor

app = FastAPI()



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Ensure the 'temp' directory exists
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Create 'temp' directory if it doesn't exist

    temp_file_path = os.path.join(temp_dir, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  # Save file to disk

    # Perform YOLO object detection
    args = dict(model='yolov8n-seg.pt', source=temp_file_path)
    predictor = SegmentationPredictor(overrides=args)
    predictor.predict_cli()
    predictor.show()

    # Optionally delete the temporary file after processing
    os.remove(temp_file_path)

    return {"filename": file.filename, "details": "Object detection performed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


