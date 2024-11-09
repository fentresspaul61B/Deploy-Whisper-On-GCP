# Deploy-Whisper-On-GCP
In this example, I will document the steps to deploy the open AI whisper SOTA STT model on GCP cloud run with Docker. 

This is useful incase you want to ensure your data is stored on specific servers you are in control of. 



## Steps

### Create new python venv
    ```
    pyenv local 3.10.0

    python -m venv venv

    source venv/bin/activate
    ```

### Install the whisper package (need to read through source code, to verify if data is being stored, and check if it can run offline)
    ```
    pip install --upgrade pip
    
    pip install -U openai-whisper

    ```

### Install FFMPEG
I already have it installed, and this is just for running the model locally, we will need to install ffmpeg within our env container as well. 

```
# From Open AI Docs: 

# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

### Additional Installs 
Read the docs from Open AI, as they suggest you may need to install the Rust programming language as well as packages like tiktoken; however, this will depend on your machine. For this demo, I am running locally on MacOS. 

### Test 
Grab any audio file with speech, to test the model locally using the CLI tool: 
```
whisper 'your_audio.mp3' --model turbo -f "json"
```
This will save your transcription data into a json file in the current dir. 

## Writing the API
First I will start by making a simple endpoint which is only used to upload audio and check the file name and its length in bytes. 

### Install Fast API
```
pip install fastapi 

pip install uvicorn 

# required for file uploading with fast API (Why?)
python-multipart
```

### Python API Endpoint
For loading the audio into our API, we will utlize the fast API upload file method: https://fastapi.tiangolo.com/reference/uploadfile/ 


```python
# API file upload code:
from fastapi import FastAPI, File, UploadFile
import subprocess


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


app = FastAPI()


@app.post("/upload-audio/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    response = {
        "filename": file.filename, 
        "content_size": len(content),
        "ffmpeg": check_ffmpeg()
    }
    return response
```

Check the api locally in development mode: 
```
fastapi dev main.py
```

Navigate to this URL to check out your API:
```
http://127.0.0.1:8000/docs
```

Next click on the "try it out" button. 

Then go to the file upload, and upload an MP3 file, and hit the "Execute" button. 

This will make a request to the API, the output should be something like so:

```json
{
  "filename": "your_audio.mp3",
  "content_size": 724320,
  "ffmpeg": true
}
```

## STT Python Code
Now that we verified that a simple API is working locally, lets add the STT python code from the example on the open AIs GitHub and add it to the API. 

```python
import whisper
from fastapi import FastAPI, File, UploadFile
import tempfile
import shutil
import os


def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, temp_file)
        temp_file_path = temp_file.name
    return temp_file_path


app = FastAPI()
MODEL = whisper.load_model("turbo")


@app.post("/translate/")
async def translate(file: UploadFile = File(...)):
    temp_filepath = save_upload_file_to_temp(file)
    audio = whisper.load_audio(temp_filepath)  # Returns np.array
    audio = whisper.pad_or_trim(audio)  # Fixes np.array shape
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(MODEL.device)
    options = whisper.DecodingOptions()  # This is empty, but a required param.
    result = whisper.decode(MODEL, mel, options)
    os.remove(temp_filepath)
    return {"text": result.text, "language": result.language}
```

How this works: 







