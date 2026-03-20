# 2_scenes_revit.py
# pip install opencv-python scikit-image
import cv2, json, os
from skimage.metrics import structural_similarity as ssim

VIDEO_PATH = r"C:\Users\charb\Downloads\video_rag_train.mp4"
FRAMES_DIR = "scene_frames"
SCENES_JSON = "scenes.json"
os.makedirs(FRAMES_DIR, exist_ok=True)

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
    cap = cv2.VideoCapture(VIDEO_PATH)
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

        # match sizes (defensive)
        h = min(prev.shape[0], cur.shape[0]); w = min(prev.shape[1], cur.shape[1])
        a = prev[:h, :w]; b = cur[:h, :w]

        score = ssim(a, b)
        # trigger a cut if the viewport changed enough & we haven't cut too recently
        if score < SSIM_THRESHOLD and (t - last_cut) >= MIN_SCENE_SECONDS:
            cut_times.append(t)
            last_cut = t
            prev = cur
        else:
            prev = cur
        t += STEP_SECONDS

    cut_times.append(duration)

    # save mid-frames & manifest
    for i in range(len(cut_times) - 1):
        s = float(cut_times[i]); e = float(cut_times[i + 1])
        mid = (s + e) / 2.0
        fp = os.path.join(FRAMES_DIR, f"scene_{i:04d}.jpg")
        save_frame(cap, mid, fp)
        scenes.append({"scene_id": i, "start": s, "end": e, "frame_path": fp})

    cap.release()

    with open(SCENES_JSON, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)

    avg = (sum((scenes[i]["end"] - scenes[i]["start"]) for i in range(len(scenes))) / max(1, len(scenes)))
    print(f"Saved {len(scenes)} scenes to {SCENES_JSON} (avg len ≈ {avg:.1f}s); frames in {FRAMES_DIR}")

if __name__ == "__main__":
    main()
