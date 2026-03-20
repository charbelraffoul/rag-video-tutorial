# 2_scenes.py
# pip install opencv-python scikit-image
import argparse, cv2, json, os
from skimage.metrics import structural_similarity as ssim

# ---- Tunables (good starting points for Revit tutorials) ----
STEP_SECONDS       = 0.5     # sample every 0.5s
MIN_SCENE_SECONDS  = 3.0     # don't cut more frequently than this
SSIM_THRESHOLD     = 0.990   # lower -> more cuts (0.985–0.995 is a sane range)
DOWNSCALE_WIDTH    = 640     # speedup; keep aspect ratio
GAUSSIAN_BLUR_KSZ  = (5, 5)  # blur to reduce cursor/tooltip noise

# ROI: keep central viewport; ignore ribbon (top) + side panels (left/right) + status bar (bottom)
KEEP_TOP_PCT    = 0.12
KEEP_BOTTOM_PCT = 0.92
KEEP_LEFT_PCT   = 0.18
KEEP_RIGHT_PCT  = 0.82
# -------------------------------------------------------------

def read_frame_at(cap, t_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None

def preprocess(frame, roi=True):
    h, w = frame.shape[:2]
    # central ROI crop
    if roi:
        y1 = int(h * KEEP_TOP_PCT);     y2 = int(h * KEEP_BOTTOM_PCT)
        x1 = int(w * KEEP_LEFT_PCT);    x2 = int(w * KEEP_RIGHT_PCT)
        frame = frame[y1:y2, x1:x2]
    # downscale (preserve aspect)
    if DOWNSCALE_WIDTH and frame.shape[1] > DOWNSCALE_WIDTH:
        scale = DOWNSCALE_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (DOWNSCALE_WIDTH, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    # grayscale + blur (reduce cursor/tooltips flicker)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KSZ, 0)
    return gray

def save_frame(cap, t_sec, out_path):
    fr = read_frame_at(cap, t_sec)
    if fr is None: return False
    cv2.imwrite(out_path, fr)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path",  required=True,                help="Path to the input video file")
    parser.add_argument("--video-id",    default=None,                 help="Used to auto-name output files (optional)")
    parser.add_argument("--frames-dir",  default=None,                 help="Directory to save frame thumbnails (default: scene_frames/<video-id> or scene_frames)")
    parser.add_argument("--scenes-json", default=None,                 help="Output JSON with scene metadata (default: scenes_<video-id>.json or scenes.json)")
    args = parser.parse_args()

    vid = args.video_id
    frames_dir  = args.frames_dir  or (f"scene_frames/{vid}" if vid else "scene_frames")
    scenes_json = args.scenes_json or (f"scenes_{vid}.json"  if vid else "scenes.json")

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if total else 0.0

    scenes = []
    t = 0.0
    prev_raw = read_frame_at(cap, t)
    if prev_raw is None:
        print("Failed to read first frame."); return
    prev = preprocess(prev_raw)
    last_cut = 0.0
    cut_times = [0.0]

    t = STEP_SECONDS
    while t < max(0.01, duration - 0.01):
        raw = read_frame_at(cap, t)
        if raw is None: break
        cur = preprocess(raw)

        h = min(prev.shape[0], cur.shape[0]); w = min(prev.shape[1], cur.shape[1])
        a = prev[:h, :w]; b = cur[:h, :w]

        score = ssim(a, b)
        if score < SSIM_THRESHOLD and (t - last_cut) >= MIN_SCENE_SECONDS:
            cut_times.append(t)
            last_cut = t
            prev = cur
        else:
            prev = cur
        t += STEP_SECONDS

    cut_times.append(duration)

    for i in range(len(cut_times) - 1):
        s = float(cut_times[i]); e = float(cut_times[i + 1])
        mid = (s + e) / 2.0
        fp = os.path.join(frames_dir, f"scene_{i:04d}.jpg")
        save_frame(cap, mid, fp)
        scenes.append({"scene_id": i, "start": s, "end": e, "frame_path": fp})

    cap.release()

    with open(scenes_json, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)

    avg = (sum((scenes[i]["end"] - scenes[i]["start"]) for i in range(len(scenes))) / max(1, len(scenes)))
    print(f"Saved {len(scenes)} scenes to {scenes_json} (avg len ≈ {avg:.1f}s); frames in {frames_dir}")

if __name__ == "__main__":
    main()
