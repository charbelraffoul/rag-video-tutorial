# 1_transcribe.py
import argparse, whisper, json, os
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_audio(video_path: str, out_audio: str):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(out_audio, fps=16000, codec='pcm_s16le')
    clip.close()


def transcribe(audio_path: str, model_size: str):
    print("Loading Whisper model…")
    model = whisper.load_model(model_size)
    print(f"Transcribing {audio_path}…")
    result = model.transcribe(audio_path, fp16=False)
    return result


def save_outputs(result, segments_jsonl: str, full_txt: str):
    with open(full_txt, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    with open(segments_jsonl, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            out = {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {segments_jsonl} and {full_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path",  required=True, help="Path to the input video file")
    parser.add_argument("--video-id",    default=None,  help="Used to auto-name output files (optional)")
    parser.add_argument("--audio-out",   default=None,  help="Where to write the extracted audio")
    parser.add_argument("--segments",    default=None,  help="Output JSONL with timestamped segments")
    parser.add_argument("--transcript",  default=None,  help="Output plain-text transcript")
    parser.add_argument("--model",       default="base", help="Whisper model size (tiny/base/small/medium/large)")
    args = parser.parse_args()

    vid = args.video_id
    audio_out  = args.audio_out  or (f"audio_{vid}.wav"                    if vid else "audio.wav")
    segments   = args.segments   or (f"transcript_segments_{vid}.jsonl"    if vid else "transcript_segments.jsonl")
    transcript = args.transcript or (f"transcript_{vid}.txt"               if vid else "transcript.txt")

    if not os.path.exists(audio_out):
        print("Extracting audio from video…")
        extract_audio(args.video_path, audio_out)
    else:
        print(f"{audio_out} already exists, skipping extraction.")

    res = transcribe(audio_out, args.model)
    save_outputs(res, segments, transcript)


if __name__ == "__main__":
    main()
