import requests
import json
import time
import os
import logging
from openai import OpenAI
from pathlib import Path
import hashlib
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcription.text


def speak_text(text):

    # Make the TTS request
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    project_root = Path(__file__).parent.parent
    audio_dir = project_root / "audio"
    audio_dir.mkdir(exist_ok=True)
    random_string = str(random.randint(1, 1000000))
    hash_object = hashlib.sha1(random_string.encode())
    filename = hash_object.hexdigest()[:8]
    file_path = audio_dir / f"{filename}.mp3"

    response.stream_to_file(file_path)
    print(f"Audio content saved as '{file_path}'")
    return file_path
