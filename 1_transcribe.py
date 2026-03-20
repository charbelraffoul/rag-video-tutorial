# 1_transcribe.py
import whisper, json, os
from moviepy.video.io.VideoFileClip import VideoFileClip

VIDEO_PATH      = r"C:\Users\charb\Downloads\video_rag_train.mp4"
AUDIO_OUTPUT    = "audio.wav"
SEGMENTS_JSONL  = "transcript_segments.jsonl"   # <- per-segment with timestamps
FULL_TXT        = "transcript.txt"

def extract_audio(video_path: str, out_audio: str):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(out_audio, fps=16000, codec='pcm_s16le')
    clip.close()

def transcribe(audio_path: str):
    print("Loading Whisper model…")
    model = whisper.load_model("base")
    print(f"Transcribing {audio_path}…")
    result = model.transcribe(audio_path, fp16=False)
    return result

def save_outputs(result):
    # full text (optional)
    with open(FULL_TXT, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    # segments JSONL (start, end, text)
    with open(SEGMENTS_JSONL, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            out = {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {SEGMENTS_JSONL} and {FULL_TXT}")

def main():
    if not os.path.exists(AUDIO_OUTPUT):
        print("Extracting audio from video…")
        extract_audio(VIDEO_PATH, AUDIO_OUTPUT)
    else:
        print(f"{AUDIO_OUTPUT} already exists, skipping extraction.")
    res = transcribe(AUDIO_OUTPUT)
    save_outputs(res)

if __name__ == "__main__":
    main()
