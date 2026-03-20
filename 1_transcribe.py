# 1_transcribe.py

import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

# Path to your video file
VIDEO_PATH      = r"C:\Users\charb\Downloads\video_rag_train.mp4"
AUDIO_OUTPUT    = "audio.wav"
TRANSCRIPT_PATH = "transcript.txt"

def extract_audio(video_path: str, out_audio: str):
    """Extracts audio track from a video into a WAV file."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(out_audio, fps=16000, codec='pcm_s16le')
    clip.close()

def transcribe(audio_path: str, transcript_path: str):
    """Uses Whisper to transcribe audio to text."""
    print(f"Loading Whisper model…")
    model = whisper.load_model("base")  # you can switch to "small","medium","large"
    print(f"Transcribing {audio_path}…")
    result = model.transcribe(audio_path, fp16=False)
    
    text = result["text"]
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcript saved to {transcript_path}")

def main():
    if not os.path.exists(AUDIO_OUTPUT):
        print("Extracting audio from video…")
        extract_audio(VIDEO_PATH, AUDIO_OUTPUT)
    else:
        print(f"{AUDIO_OUTPUT} already exists, skipping extraction.")
    
    transcribe(AUDIO_OUTPUT, TRANSCRIPT_PATH)

if __name__ == "__main__":
    main()
