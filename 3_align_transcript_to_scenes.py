# 3_align_transcript_to_scenes.py
import json, os

SCENES_JSON = "scenes.json"                  # your file above
SEGMENTS_JSONL = "transcript_segments.jsonl" # from Whisper (per-segment timestamps)
OUT_JSONL = "video_manifest.jsonl"
MIN_OVERLAP_SECONDS = 0.20  # keep a segment if it overlaps a scene by >= 200 ms

def load_scenes(path):
    with open(path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    scenes.sort(key=lambda x: (x["start"], x["end"]))
    return scenes

def load_segments(path):
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                segs.append(json.loads(line))
    segs.sort(key=lambda x: (x["start"], x["end"]))
    return segs

def overlap(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))

def main():
    if not os.path.exists(SEGMENTS_JSONL):
        raise FileNotFoundError("Missing transcript_segments.jsonl. Re-run Whisper saving segments.")
    scenes = load_scenes(SCENES_JSON)
    segs = load_segments(SEGMENTS_JSONL)

    j = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for sc in scenes:
            s, e = float(sc["start"]), float(sc["end"])
            # advance cursor to first possibly-overlapping segment
            while j < len(segs) and segs[j]["end"] <= s:
                j += 1
            k = j
            attached = []
            while k < len(segs) and segs[k]["start"] < e:
                sg = segs[k]
                ov = overlap(sg["start"], sg["end"], s, e)
                if ov >= MIN_OVERLAP_SECONDS:
                    attached.append({
                        "start": sg["start"], "end": sg["end"],
                        "text": sg["text"].strip(),
                        "overlap_seconds": ov
                    })
                k += 1

            attached.sort(key=lambda x: (x["start"], x["end"]))
            merged = " ".join(x["text"] for x in attached).strip()

            out.write(json.dumps({
                "scene_id": sc["scene_id"],
                "scene_start": s,
                "scene_end": e,
                "duration": round(e - s, 3),
                "frame_path": sc["frame_path"],
                "text": merged,
                "segments": attached
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_JSONL} with {len(scenes)} rows.")

if __name__ == "__main__":
    main()
