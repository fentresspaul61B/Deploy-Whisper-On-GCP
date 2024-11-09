import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
import subprocess
import tempfile
import shutil
import os
import torch

# POSSIBLE VERSIONS
# https://github.com/openai/whisper/tree/main
# 'tiny.en', 'tiny', 'base.en', 'base', 'small.en',
# 'small', 'medium.en', 'medium', 'large-v1', 'large-v2',
# 'large-v3', 'large', 'large-v3-turbo', 'turbo'
MODEL_VERSION = "tiny"
# V3 models require 128, other models require 80
NUM_MELS = 80

app = FastAPI()
MODEL = whisper.load_model(MODEL_VERSION)


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


@app.post("/translate/")
async def translate(file: UploadFile = File(...)):
    temp_filepath = save_upload_file_to_temp(file)
    try:
        audio = whisper.load_audio(temp_filepath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(
            audio, n_mels=NUM_MELS).to(MODEL.device)
        result = whisper.decode(MODEL, mel)
    finally:
        os.remove(temp_filepath)
    return {"text": result.text, "language": result.language}
