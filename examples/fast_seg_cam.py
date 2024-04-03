import os
import shutil
from fastapi import FastAPI, UploadFile, Query,File
import uvicorn
from enum import Enum
from typing import Optional
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from dotenv import load_dotenv
load_dotenv()  # This loads the environment variables from a .env file.
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(camera: str = Query(default="none", enum=["local", "cctv", "none"])):
    # Determine the source based on the user's choice
    if camera == "cctv":
        source = os.getenv("VERTEX_CCTV_ACCESS")  # CCTV camera source
    elif camera == "local":
        source = "0"  # Local webcam source
    else:
        source = "0"  # Default case, can be adjusted as needed

    # Perform YOLO object detection
    args = dict(model='yolov8n-seg.pt', source=source)
    predictor = SegmentationPredictor(overrides=args)
    predictor.predict_cli()

    # Return a response indicating the chosen source
    return {"details": f"Object detection performed using {camera} camera."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)