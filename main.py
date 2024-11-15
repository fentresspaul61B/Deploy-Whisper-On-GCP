import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
import subprocess
import tempfile
import shutil
import os
import torch
import time

# POSSIBLE VERSIONS
# https://github.com/openai/whisper/tree/main
# 'tiny.en', 'tiny', 'base.en', 'base', 'small.en',
# 'small', 'medium.en', 'medium', 'large-v1', 'large-v2',
# 'large-v3', 'large', 'large-v3-turbo', 'turbo'
MODEL_VERSION = "large-v3-turbo"

# V3 models require 128 mel, other models like the tiny model require 80 mels
NUM_MELS = 128

app = FastAPI()

# Model loaded in docker file.
MODEL_PATH = f"/app/models/{MODEL_VERSION}.pt"
MODEL = whisper.load_model(MODEL_PATH)


def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, temp_file)
        temp_file_path = temp_file.name
    return temp_file_path


@app.post("/check-gpu/")
async def check_gpu():
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA is not available")
    return {"cuda": True}


@app.post("/check-ffmpeg/")
async def check_ffmpeg():
    ffmpeg = True
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception as e:
        print(e)
        ffmpeg = False
    if not ffmpeg:
        raise HTTPException(status_code=400, detail="FFMPEG is not available")
    return {"ffmpeg": True}


@app.post("/check-model-in-memory/")
async def check_model_in_memory():
    """Verifies if model was loaded during docker build."""
    return {"contents": os.listdir("/app/models/")}


@app.post("/translate/")
async def translate(file: UploadFile = File(...)):
    result = {}

    s = time.time()
    temp_filepath = save_upload_file_to_temp(file)
    e = time.time()
    result["temp_file_time"] = e - s

    try:
        s = time.time()
        audio = whisper.load_audio(temp_filepath)
        e = time.time()
        result["load_audio_time"] = e - s

        s = time.time()
        audio = whisper.pad_or_trim(audio)
        e = time.time()
        result["pad_audio_time"] = e - s

        s = time.time()
        mel = whisper.log_mel_spectrogram(
            audio, n_mels=NUM_MELS).to(MODEL.device)
        e = time.time()
        result["compute_mel_features_time"] = e - s

        s = time.time()
        result = whisper.decode(MODEL, mel)
        e = time.time()
        result["inference_time"] = e - s
    finally:
        os.remove(temp_filepath)
    result["text"] = result.text
    result["language"] = result.language
    return result
