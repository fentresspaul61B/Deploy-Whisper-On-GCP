import whisper
from fastapi import FastAPI, File, UploadFile
# import subprocess
import tempfile
import shutil
import os


# def check_ffmpeg():
#     try:
#         subprocess.run(
#             ["ffmpeg", "-version"],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         return True
#     except subprocess.CalledProcessError:
#         return False
#     except FileNotFoundError:
#         return False


def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, temp_file)
        temp_file_path = temp_file.name
    return temp_file_path


app = FastAPI()
MODEL = whisper.load_model("turbo")


# @app.post("/upload-audio/")
# async def create_upload_file(file: UploadFile = File(...)):
#     content = await file.read()
#     response = {
#         "filename": file.filename,
#         "content_size": len(content),
#         "ffmpeg": check_ffmpeg()
#     }
#     return response


# @app.post("/check-gpu/")
# async def check_gpu():
#     "cuda" if torch.cuda.is_available() else "cpu"


@app.post("/translate/")
async def translate(file: UploadFile = File(...)):
    temp_filepath = save_upload_file_to_temp(file)
    try:
        audio = whisper.load_audio(temp_filepath)  # Returns np.array
        audio = whisper.pad_or_trim(audio)  # Fixes np.array shape
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(MODEL.device)
        result = whisper.decode(MODEL, mel)
    finally:
        os.remove(temp_filepath)  # Makes sure file is always deleted.
    return {"text": result.text, "language": result.language}
