ls -lah face_recognition_sface.onnx
python - <<'PY'
import cv2 as cv
p="/home/pi/project/models/face_recognition_sface.onnx"
net = cv.dnn.readNetFromONNX(p)
print("Loaded SFace OK")
PY

cd ..
source ~/fr-venv/bin/activate
pip install --upgrade numpy
clear
cat > ~/project/app.py <<'PY'
#!/usr/bin/env python3
from flask import Flask, Response, request, jsonify
import cv2 as cv
import numpy as np
import os, time, json, subprocess
from threading import Lock

APP_HOST = "0.0.0.0"
APP_PORT = 5000

BASE_DIR = "/home/pi/project"
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "db")
USERS_JSON = os.path.join(DB_DIR, "users.json")

MODEL_YUNET = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
MODEL_LIVE  = os.path.join(MODEL_DIR, "liveness.onnx")
MODEL_SFACE = os.path.join(MODEL_DIR, "face_recognition_sface.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def download(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = subprocess.run(["curl", "-L", "-o", out_path, url],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {out_path}:\n{r.stderr}")

def ensure_model(path: str, url: str, validate_fn):
    if not os.path.exists(path):
        print("[INFO] downloading", path)
        download(url, path)
    try:
        validate_fn(path)
        return
    except Exception:
        # redownload once
        try: os.remove(path)
        except Exception: pass
        print("[INFO] re-downloading", path)
        download(url, path)
        validate_fn(path)

def validate_onnx_loadable(path: str):
    _ = cv.dnn.readNetFromONNX(path)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def yaw_from_landmarks(lmk_full):
    lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry = [float(v) for v in lmk_full[:10]]
    eye_mid_x = (lx + rx) * 0.5
    eye_dist = abs(rx - lx) + 1e-6
    return (nx - eye_mid_x) / eye_dist

def load_users():
    os.makedirs(DB_DIR, exist_ok=True)
    if not os.path.exists(USERS_JSON):
        return {}
    with open(USERS_JSON, "r") as f:
        return json.load(f)

def save_users(users):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(USERS_JSON, "w") as f:
        json.dump(users, f, indent=2)

def save_user_embedding(name: str, emb_list):
    # store multiple embeddings + their mean
    embs = np.stack(emb_list).astype(np.float32)
    mean = embs.mean(axis=0)
    out = os.path.join(DB_DIR, f"{name}.npz")
    np.savez_compressed(out, embs=embs, mean=mean)

def load_user_embedding(name: str):
    path = os.path.join(DB_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["mean"].astype(np.float32)

# OpenCV tuning
try:
    cv.setUseOptimized(True)
    cv.setNumThreads(0)
except Exception:
    pass

# Ensure models
ensure_model(MODEL_YUNET, YUNET_URL, validate_onnx_loadable)
ensure_model(MODEL_SFACE, SFACE_URL, validate_onnx_loadable)

if not (os.path.exists(MODEL_LIVE) and os.path.getsize(MODEL_LIVE) > 100_000):
    raise RuntimeError(f"Missing liveness model: {MODEL_LIVE}")

print("[INFO] models OK")

# Load detectors/recognizer
detector = cv.FaceDetectorYN.create(MODEL_YUNET, "", (320, 320), 0.6, 0.3, 5000)
recognizer = cv.FaceRecognizerSF.create(MODEL_SFACE, "")  # SFace
live_net = cv.dnn.readNetFromONNX(MODEL_LIVE)

# Camera
cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

# Calibrate liveness class index
real_idx = None
calib_logits = []
t0 = time.time()
while time.time() - t0 < 1.5:
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]
    small = cv.resize(frame, (320, 180), interpolation=cv.INTER_LINEAR)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        continue
    f = max(faces, key=lambda x: x[2] * x[3])
    x, y, bw, bh = f[:4].astype(int)

    sx = w / small.shape[1]
    sy = h / small.shape[0]
    x1 = int(x * sx); y1 = int(y * sy)
    x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)
    pad = int(0.20 * max(x2 - x1, y2 - y1))
    x1 = clamp(x1 - pad, 0, w - 1); y1 = clamp(y1 - pad, 0, h - 1)
    x2 = clamp(x2 + pad, 0, w - 1); y2 = clamp(y2 + pad, 0, h - 1)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        continue
    blob = cv.dnn.blobFromImage(face, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    live_net.setInput(blob)
    out = live_net.forward()
    logits = out.reshape(-1).astype(np.float32)
    if logits.size >= 2:
        calib_logits.append(logits[:2])

if calib_logits:
    mean_logits = np.mean(np.stack(calib_logits), axis=0)
    real_idx = int(np.argmax(mean_logits))
print("[INFO] real_idx =", real_idx)

# Active liveness state
last_left_time = 0.0
last_right_time = 0.0
CHALLENGE_WINDOW = 4.0
YAW_LEFT  = -0.18
YAW_RIGHT = +0.18

# Runtime state (shared)
lock = Lock()
state = {
    "enrolling": False,
    "enroll_name": None,
    "enroll_count": 0,
    "enroll_target": 30,
    "last_live_prob": None,
    "last_yaw": None,
    "last_status": "idle",
    "last_match": None,
    "last_match_score": None,
}

def is_blurry(img, thresh=60.0):
    # variance of Laplacian
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    v = cv.Laplacian(g, cv.CV_64F).var()
    return v < thresh

def get_face_and_landmarks(frame):
    h, w = frame.shape[:2]
    small = cv.resize(frame, (320, 180), interpolation=cv.INTER_LINEAR)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        return None

    f = max(faces, key=lambda x: x[2] * x[3])
    x, y, bw, bh = f[:4].astype(int)
    lmk = f[4:14]  # 10 values

    sx = w / small.shape[1]
    sy = h / small.shape[0]

    x1 = int(x * sx); y1 = int(y * sy)
    x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)

    pad = int(0.20 * max(x2 - x1, y2 - y1))
    x1 = clamp(x1 - pad, 0, w - 1); y1 = clamp(y1 - pad, 0, h - 1)
    x2 = clamp(x2 + pad, 0, w - 1); y2 = clamp(y2 + pad, 0, h - 1)

    # landmarks scaled to full-res
    lmk_full = []
    for i in range(0, 10, 2):
        lmk_full.append(float(lmk[i]) * sx)
        lmk_full.append(float(lmk[i+1]) * sy)

    return (x1, y1, x2, y2, np.array(lmk_full, dtype=np.float32))

def liveness_score(face_bgr):
    blob = cv.dnn.blobFromImage(face_bgr, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    live_net.setInput(blob)
    out = live_net.forward()
    logits = out.reshape(-1).astype(np.float32)[:2]
    p = softmax(logits)
    return float(p[real_idx]) if real_idx is not None else float(np.max(p))

def face_embedding(frame, bbox, landmarks_full):
    # Align face using SFace helper
    # recognizer.alignCrop expects face info in the same format returned by detector.detect on ORIGINAL frame.
    # We'll build a face array: [x,y,w,h, lmk(10)]
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    face_info = np.array([x1,y1,w,h, *landmarks_full.tolist()], dtype=np.float32)
    aligned = recognizer.alignCrop(frame, face_info)
    feat = recognizer.feature(aligned)
    # feature returns (1, 128) float32
    return feat.flatten().astype(np.float32)

app = Flask(__name__)

@app.route("/status")
def status():
    with lock:
        return jsonify(state)

@app.route("/enroll_start", methods=["POST"])
def enroll_start():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "missing name"}), 400

    with lock:
        state["enrolling"] = True
        state["enroll_name"] = name
        state["enroll_count"] = 0
        state["last_status"] = f"enrolling {name}"
    return jsonify({"ok": True, "name": name})

@app.route("/enroll_stop", methods=["POST"])
def enroll_stop():
    with lock:
        state["enrolling"] = False
        state["enroll_name"] = None
        state["last_status"] = "enroll stopped"
    return jsonify({"ok": True})

def gen_frames():
    global last_left_time, last_right_time

    PASSIVE_THRESH = 0.92
    LIVE_HITS_REQUIRED = 5
    hits = 0

    users = load_users()
    known = {name: load_user_embedding(name) for name in users.keys()}

    enroll_samples = []

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        now = time.time()
        found = get_face_and_landmarks(frame)

        status = "NO FACE"
        live_prob = None
        yaw = None
        match_name = None
        match_score = None

        if found is not None:
            x1,y1,x2,y2,lmk_full = found
            bbox = (x1,y1,x2,y2)
            face = frame[y1:y2, x1:x2]

            if face.size != 0 and (x2-x1) >= 160 and not is_blurry(face, 60.0):
                live_prob = liveness_score(face)
                yaw = yaw_from_landmarks(lmk_full)

                # update active challenge
                if yaw <= YAW_LEFT:
                    last_left_time = now
                if yaw >= YAW_RIGHT:
                    last_right_time = now
                left_ok = (now - last_left_time) <= CHALLENGE_WINDOW
                right_ok = (now - last_right_time) <= CHALLENGE_WINDOW
                active_ok = left_ok and right_ok
                passive_ok = (live_prob >= PASSIVE_THRESH)

                if passive_ok and active_ok:
                    hits += 1
                else:
                    hits = max(0, hits - 1)

                live_ok = hits >= LIVE_HITS_REQUIRED

                if not passive_ok:
                    status = "SPOOF? (PASSIVE FAIL)"
                elif not active_ok:
                    status = "TURN HEAD L/R"
                elif not live_ok:
                    status = "HOLD..."
                else:
                    status = "LIVE OK"

                # If live OK: do recognition
                if live_ok:
                    emb = face_embedding(frame, bbox, lmk_full)

                    # verify vs known users
                    best_name = None
                    best_sim = -1.0
                    for name, ref in known.items():
                        if ref is None:
                            continue
                        sim = cosine_sim(emb, ref)
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name

                    # Threshold depends on environment; start conservative
                    RECOG_THRESH = 0.35
                    if best_name is not None and best_sim >= RECOG_THRESH:
                        match_name = best_name
                        match_score = best_sim
                        status = f"VERIFIED: {best_name}"
                    else:
                        match_name = None
                        match_score = best_sim if best_sim > -1 else None
                        status = "UNKNOWN"

                    # Enrollment collection
                    with lock:
                        enrolling = state["enrolling"]
                        enroll_name = state["enroll_name"]
                        enroll_target = state["enroll_target"]

                    if enrolling and enroll_name:
                        enroll_samples.append(emb)
                        with lock:
                            state["enroll_count"] = len(enroll_samples)
                            state["last_status"] = f"enrolling {enroll_name} ({len(enroll_samples)}/{enroll_target})"

                        if len(enroll_samples) >= enroll_target:
                            # Save
                            save_user_embedding(enroll_name, enroll_samples)
                            users = load_users()
                            users[enroll_name] = {"created": time.time(), "count": len(enroll_samples)}
                            save_users(users)
                            known[enroll_name] = load_user_embedding(enroll_name)
                            enroll_samples = []

                            with lock:
                                state["enrolling"] = False
                                state["enroll_name"] = None
                                state["last_status"] = f"enrolled {enroll_name}"

            # draw bbox + info
            live_ok_draw = (hits >= LIVE_HITS_REQUIRED)
            color = (0,255,0) if live_ok_draw else (0,255,255) if (live_prob is not None and live_prob >= PASSIVE_THRESH) else (0,0,255)
            cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            ytxt = max(30, y1 - 10)
            if live_prob is not None:
                cv.putText(frame, f"live={live_prob:.3f}", (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                ytxt += 26
            if yaw is not None:
                cv.putText(frame, f"yaw={yaw:+.3f}", (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                ytxt += 26
            if match_score is not None and match_name is not None:
                cv.putText(frame, f"{match_name} sim={match_score:.3f}", (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                ytxt += 26
            elif match_score is not None and match_name is None:
                cv.putText(frame, f"best_sim={match_score:.3f}", (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                ytxt += 26
            cv.putText(frame, status, (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        with lock:
            state["last_live_prob"] = live_prob
            state["last_yaw"] = yaw
            state["last_status"] = status
            state["last_match"] = match_name
            state["last_match_score"] = match_score

        ok, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def index():
    return """
    <html><head><title>Pi Face Verify</title></head>
    <body>
      <h3>Pi Face Verify (YuNet + Liveness + SFace)</h3>
      <p>Enroll: POST /enroll_start?name=YOURNAME (then look at camera until it finishes)</p>
      <p>Stop:   POST /enroll_stop</p>
      <img src="/video" style="max-width: 100%;"/>
    </body></html>
    """

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, threaded=True)
PY

clear
cd ~/project
source ~/fr-venv/bin/activate
python app.py
clear
cat > ~/project/app.py <<'PY'
#!/usr/bin/env python3
from flask import Flask, Response, request, jsonify
import cv2 as cv
import numpy as np
import os, time, json, subprocess
from threading import Lock

APP_HOST = "0.0.0.0"
APP_PORT = 5000

BASE_DIR = "/home/pi/project"
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "db")
USERS_JSON = os.path.join(DB_DIR, "users.json")

MODEL_YUNET = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
MODEL_LIVE  = os.path.join(MODEL_DIR, "liveness.onnx")
MODEL_SFACE = os.path.join(MODEL_DIR, "face_recognition_sface.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# ------------------ utils ------------------
def download(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = subprocess.run(["curl", "-L", "-o", out_path, url],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {out_path}:\n{r.stderr}")

def validate_onnx_loadable(path: str):
    _ = cv.dnn.readNetFromONNX(path)

def ensure_model(path: str, url: str):
    if not os.path.exists(path):
        print("[INFO] downloading", path)
        download(url, path)
    try:
        validate_onnx_loadable(path)
        return
    except Exception:
        # redownload once
        try: os.remove(path)
        except Exception: pass
        print("[INFO] re-downloading", path)
        download(url, path)
        validate_onnx_loadable(path)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def yaw_from_landmarks(lmk_full):
    # (x,y)*5: left_eye, right_eye, nose, mouth_l, mouth_r
    lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry = [float(v) for v in lmk_full[:10]]
    eye_mid_x = (lx + rx) * 0.5
    eye_dist = abs(rx - lx) + 1e-6
    return (nx - eye_mid_x) / eye_dist

def is_blurry(img, thresh=60.0):
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(g, cv.CV_64F).var() < thresh

def load_users():
    os.makedirs(DB_DIR, exist_ok=True)
    if not os.path.exists(USERS_JSON):
        return {}
    with open(USERS_JSON, "r") as f:
        return json.load(f)

def save_users(users):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(USERS_JSON, "w") as f:
        json.dump(users, f, indent=2)

def save_user_embedding(name: str, emb_list):
    embs = np.stack(emb_list).astype(np.float32)
    mean = embs.mean(axis=0)
    out = os.path.join(DB_DIR, f"{name}.npz")
    np.savez_compressed(out, embs=embs, mean=mean)

def load_user_embedding(name: str):
    path = os.path.join(DB_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["mean"].astype(np.float32)

# ------------------ OpenCV tuning ------------------
try:
    cv.setUseOptimized(True)
    cv.setNumThreads(0)
except Exception:
    pass

# ------------------ Ensure models ------------------
ensure_model(MODEL_YUNET, YUNET_URL)
ensure_model(MODEL_SFACE, SFACE_URL)
if not (os.path.exists(MODEL_LIVE) and os.path.getsize(MODEL_LIVE) > 100_000):
    raise RuntimeError(f"Missing liveness model: {MODEL_LIVE}")

print("[INFO] Models OK")

# ------------------ Load models ------------------
detector = cv.FaceDetectorYN.create(MODEL_YUNET, "", (320, 320), 0.6, 0.3, 5000)
recognizer = cv.FaceRecognizerSF.create(MODEL_SFACE, "")
live_net = cv.dnn.readNetFromONNX(MODEL_LIVE)

# ------------------ Camera ------------------
cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

# ------------------ Calibrate liveness "real" index ------------------
real_idx = None
calib_logits = []
t0 = time.time()
while time.time() - t0 < 1.5:
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]
    small = cv.resize(frame, (320, 180), interpolation=cv.INTER_LINEAR)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        continue

    f = max(faces, key=lambda x: x[2] * x[3])
    x, y, bw, bh = f[:4].astype(int)

    sx = w / small.shape[1]
    sy = h / small.shape[0]
    x1 = int(x * sx); y1 = int(y * sy)
    x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)

    pad = int(0.20 * max(x2 - x1, y2 - y1))
    x1 = clamp(x1 - pad, 0, w - 1); y1 = clamp(y1 - pad, 0, h - 1)
    x2 = clamp(x2 + pad, 0, w - 1); y2 = clamp(y2 + pad, 0, h - 1)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        continue

    blob = cv.dnn.blobFromImage(face, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    live_net.setInput(blob)
    out = live_net.forward()
    logits = out.reshape(-1).astype(np.float32)
    if logits.size >= 2:
        calib_logits.append(logits[:2])

if calib_logits:
    mean_logits = np.mean(np.stack(calib_logits), axis=0)
    real_idx = int(np.argmax(mean_logits))
print("[INFO] real_idx =", real_idx)

# ------------------ Active challenge state ------------------
last_left_time = 0.0
last_right_time = 0.0
CHALLENGE_WINDOW = 4.0
YAW_LEFT  = -0.18
YAW_RIGHT = +0.18

# ------------------ Shared state for UI ------------------
lock = Lock()
state = {
    "enrolling": False,
    "enroll_name": None,
    "enroll_count": 0,
    "enroll_target": 30,
    "last_live_prob": None,
    "last_yaw": None,
    "last_status": "idle",
    "last_badge": "gray",     # green/yellow/red/gray
    "last_match": None,
    "last_match_score": None,
    "known_users": [],
}

def get_face_and_landmarks(frame):
    h, w = frame.shape[:2]
    small = cv.resize(frame, (320, 180), interpolation=cv.INTER_LINEAR)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        return None

    f = max(faces, key=lambda x: x[2] * x[3])
    x, y, bw, bh = f[:4].astype(int)
    lmk = f[4:14]  # 10 values

    sx = w / small.shape[1]
    sy = h / small.shape[0]

    x1 = int(x * sx); y1 = int(y * sy)
    x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)

    pad = int(0.20 * max(x2 - x1, y2 - y1))
    x1 = clamp(x1 - pad, 0, w - 1); y1 = clamp(y1 - pad, 0, h - 1)
    x2 = clamp(x2 + pad, 0, w - 1); y2 = clamp(y2 + pad, 0, h - 1)

    lmk_full = []
    for i in range(0, 10, 2):
        lmk_full.append(float(lmk[i]) * sx)
        lmk_full.append(float(lmk[i+1]) * sy)

    return (x1, y1, x2, y2, np.array(lmk_full, dtype=np.float32))

def liveness_score(face_bgr):
    blob = cv.dnn.blobFromImage(face_bgr, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    live_net.setInput(blob)
    out = live_net.forward()
    logits = out.reshape(-1).astype(np.float32)[:2]
    p = softmax(logits)
    return float(p[real_idx]) if real_idx is not None else float(np.max(p))

def face_embedding(frame, bbox, landmarks_full):
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    face_info = np.array([x1,y1,w,h, *landmarks_full.tolist()], dtype=np.float32)
    aligned = recognizer.alignCrop(frame, face_info)
    feat = recognizer.feature(aligned)  # (1, 128)
    return feat.flatten().astype(np.float32)

# ------------------ Flask ------------------
app = Flask(__name__)

HOME_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pi Face System</title>
<style>
  body { font-family: system-ui, Arial; margin: 20px; }
  .row { display:flex; gap:20px; flex-wrap:wrap; }
  .card { border:1px solid #ddd; border-radius:12px; padding:16px; }
  .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; color:#fff; }
  .green { background:#1f9d55; }
  .yellow{ background:#d97706; }
  .red   { background:#dc2626; }
  .gray  { background:#6b7280; }
  .btn { display:inline-block; padding:10px 14px; border-radius:10px; border:1px solid #ddd; background:#f8fafc; cursor:pointer; text-decoration:none; color:#111; }
  .btn:hover { background:#eef2f7; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
</head>
<body>
  <h2>Pi Face Recognition</h2>

  <div class="row">
    <div class="card">
      <div style="display:flex; align-items:center; gap:10px;">
        <div id="badge" class="badge gray">IDLE</div>
        <div class="mono" id="statusText">loading...</div>
      </div>
      <div style="margin-top:10px">
        <div>Live: <span class="mono" id="liveProb">-</span></div>
        <div>Yaw: <span class="mono" id="yaw">-</span></div>
        <div>Match: <span class="mono" id="match">-</span></div>
        <div>Users: <span class="mono" id="users">-</span></div>
      </div>
      <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
        <a class="btn" href="/enroll">Go to Enrollment</a>
        <button class="btn" onclick="refreshNow()">Refresh</button>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Live Feed</h3>
      <img src="/video" style="max-width:900px; width:100%; border-radius:10px;"/>
      <div style="margin-top:8px; color:#555; font-size:14px;">
        Tip: If you test with a phone replay, it should fail the head-turn challenge.
      </div>
    </div>
  </div>

<script>
async function refreshNow(){
  const r = await fetch('/api/status');
  const s = await r.json();

  const badge = document.getElementById('badge');
  badge.className = 'badge ' + (s.last_badge || 'gray');
  badge.textContent = (s.last_badge || 'gray').toUpperCase();

  document.getElementById('statusText').textContent = s.last_status || '-';
  document.getElementById('liveProb').textContent = (s.last_live_prob==null) ? '-' : s.last_live_prob.toFixed(3);
  document.getElementById('yaw').textContent = (s.last_yaw==null) ? '-' : s.last_yaw.toFixed(3);
  document.getElementById('match').textContent = (s.last_match==null) ? '-' : (s.last_match + ' (' + (s.last_match_score?.toFixed?.(3) ?? '-') + ')');
  document.getElementById('users').textContent = (s.known_users && s.known_users.length) ? s.known_users.join(', ') : '-';
}
setInterval(refreshNow, 500);
refreshNow();
</script>
</body>
</html>
"""

ENROLL_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Enroll</title>
<style>
  body { font-family: system-ui, Arial; margin: 20px; }
  .row { display:flex; gap:20px; flex-wrap:wrap; }
  .card { border:1px solid #ddd; border-radius:12px; padding:16px; }
  .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; color:#fff; }
  .green { background:#1f9d55; }
  .yellow{ background:#d97706; }
  .red   { background:#dc2626; }
  .gray  { background:#6b7280; }
  .btn { padding:10px 14px; border-radius:10px; border:1px solid #ddd; background:#f8fafc; cursor:pointer; }
  .btn:hover { background:#eef2f7; }
  input { padding:10px; border-radius:10px; border:1px solid #ddd; width:260px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .bar { width: 100%; height: 16px; background:#eee; border-radius: 999px; overflow:hidden; }
  .fill { height: 100%; width: 0%; background:#1f9d55; transition: width 0.2s ease; }
</style>
</head>
<body>
  <h2>Enrollment</h2>
  <div style="margin-bottom:12px;">
    <a href="/" class="btn" style="text-decoration:none; color:#111;">← Back</a>
  </div>

  <div class="row">
    <div class="card" style="min-width:340px;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div id="badge" class="badge gray">IDLE</div>
        <div class="mono" id="statusText">loading...</div>
      </div>

      <div style="margin-top:10px;">
        <div>Enrolling: <span class="mono" id="enrolling">-</span></div>
        <div>Progress: <span class="mono" id="progressText">0/0</span></div>
        <div class="bar" style="margin-top:8px;"><div id="fill" class="fill"></div></div>
      </div>

      <hr style="margin:16px 0; border:none; border-top:1px solid #eee;">

      <form id="enrollForm">
        <label>Name</label><br>
        <input id="name" placeholder="e.g. alice" required>
        <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
          <button class="btn" type="submit">Start Enrollment</button>
          <button class="btn" type="button" onclick="stopEnroll()">Stop</button>
        </div>
      </form>

      <div style="margin-top:14px; color:#555; font-size:14px;">
        Keep your face inside the box and follow the head-turn prompts.
        Enrollment auto-stops when complete.
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Live Feed</h3>
      <img src="/video" style="max-width:900px; width:100%; border-radius:10px;"/>
    </div>
  </div>

<script>
async function api(url, opts){
  const r = await fetch(url, opts);
  const j = await r.json();
  if(!r.ok) throw new Error(j.error || 'request failed');
  return j;
}

document.getElementById('enrollForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = document.getElementById('name').value.trim();
  if(!name) return;

  try{
    await api('/api/enroll/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({name})
    });
  }catch(err){
    alert(err.message);
  }
});

async function stopEnroll(){
  try{
    await api('/api/enroll/stop', {method:'POST'});
  }catch(err){
    alert(err.message);
  }
}

function setBadge(color, text){
  const badge = document.getElementById('badge');
  badge.className = 'badge ' + (color || 'gray');
  badge.textContent = (text || 'IDLE');
}

async function refresh(){
  const s = await (await fetch('/api/status')).json();

  setBadge(s.last_badge, (s.last_badge || 'gray').toUpperCase());
  document.getElementById('statusText').textContent = s.last_status || '-';

  document.getElementById('enrolling').textContent = s.enrolling ? (s.enroll_name || '-') : 'no';
  const c = s.enroll_count || 0;
  const t = s.enroll_target || 0;
  document.getElementById('progressText').textContent = `${c}/${t}`;

  const pct = t ? Math.min(100, Math.round((c / t) * 100)) : 0;
  document.getElementById('fill').style.width = pct + '%';

  // color bar based on badge
  const fill = document.getElementById('fill');
  fill.style.background = (s.last_badge === 'green') ? '#1f9d55'
                      : (s.last_badge === 'yellow') ? '#d97706'
                      : (s.last_badge === 'red') ? '#dc2626'
                      : '#6b7280';
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return HOME_HTML

@app.route("/enroll")
def enroll_page():
    return ENROLL_HTML

@app.route("/api/status")
def api_status():
    users = load_users()
    with lock:
        state["known_users"] = sorted(list(users.keys()))
        return jsonify(state)

@app.route("/api/enroll/start", methods=["POST"])
def api_enroll_start():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name is required"}), 400

    with lock:
        state["enrolling"] = True
        state["enroll_name"] = name
        state["enroll_count"] = 0
        state["last_status"] = f"enrolling {name}"
    return jsonify({"ok": True, "name": name})

@app.route("/api/enroll/stop", methods=["POST"])
def api_enroll_stop():
    with lock:
        state["enrolling"] = False
        state["enroll_name"] = None
        state["enroll_count"] = 0
        state["last_status"] = "enrollment stopped"
    return jsonify({"ok": True})

def gen_frames():
    global last_left_time, last_right_time

    # tune here
    PASSIVE_THRESH = 0.92
    LIVE_HITS_REQUIRED = 5
    RECOG_THRESH = 0.35
    ENROLL_TARGET = 30

    hits = 0
    enroll_samples = []

    users = load_users()
    known = {name: load_user_embedding(name) for name in users.keys()}

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        now = time.time()
        found = get_face_and_landmarks(frame)

        status = "NO FACE"
        badge = "gray"
        live_prob = None
        yaw = None
        match_name = None
        match_score = None

        if found is not None:
            x1,y1,x2,y2,lmk_full = found
            bbox = (x1,y1,x2,y2)
            face = frame[y1:y2, x1:x2]

            if face.size != 0 and (x2-x1) >= 160 and not is_blurry(face, 60.0):
                live_prob = liveness_score(face)
                yaw = yaw_from_landmarks(lmk_full)

                # update active challenge (left/right)
                if yaw <= YAW_LEFT:
                    last_left_time = now
                if yaw >= YAW_RIGHT:
                    last_right_time = now

                left_ok = (now - last_left_time) <= CHALLENGE_WINDOW
                right_ok = (now - last_right_time) <= CHALLENGE_WINDOW
                active_ok = left_ok and right_ok
                passive_ok = (live_prob >= PASSIVE_THRESH)

                if passive_ok and active_ok:
                    hits += 1
                else:
                    hits = max(0, hits - 1)

                live_ok = hits >= LIVE_HITS_REQUIRED

                if not passive_ok:
                    status = "SPOOF? (PASSIVE FAIL)"
                    badge = "red"
                elif not active_ok:
                    status = "TURN HEAD L/R"
                    badge = "yellow"
                elif not live_ok:
                    status = "HOLD..."
                    badge = "yellow"
                else:
                    status = "LIVE OK"
                    badge = "green"

                # recognition + enrollment only when live_ok
                if live_ok:
                    emb = face_embedding(frame, bbox, lmk_full)

                    # verify best
                    best_name = None
                    best_sim = -1.0
                    for name, ref in known.items():
                        if ref is None:
                            continue
                        sim = cosine_sim(emb, ref)
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name

                    if best_name is not None and best_sim >= RECOG_THRESH:
                        match_name = best_name
                        match_score = best_sim
                        status = f"VERIFIED: {best_name}"
                        badge = "green"
                    else:
                        match_score = best_sim if best_sim > -1 else None
                        status = "UNKNOWN"
                        badge = "yellow"

                    # enrollment capture
                    with lock:
                        enrolling = state["enrolling"]
                        enroll_name = state["enroll_name"]

                    if enrolling and enroll_name:
                        enroll_samples.append(emb)
                        with lock:
                            state["enroll_target"] = ENROLL_TARGET
                            state["enroll_count"] = len(enroll_samples)
                            state["last_status"] = f"enrolling {enroll_name} ({len(enroll_samples)}/{ENROLL_TARGET})"
                            state["last_badge"] = "yellow"

                        if len(enroll_samples) >= ENROLL_TARGET:
                            save_user_embedding(enroll_name, enroll_samples)
                            users = load_users()
                            users[enroll_name] = {"created": time.time(), "count": len(enroll_samples)}
                            save_users(users)
                            known[enroll_name] = load_user_embedding(enroll_name)
                            enroll_samples = []

                            with lock:
                                state["enrolling"] = False
                                state["enroll_name"] = None
                                state["last_status"] = f"enrolled {enroll_name}"
                                state["last_badge"] = "green"

            # draw
            color = (0,255,0) if badge=="green" else (0,255,255) if badge=="yellow" else (0,0,255) if badge=="red" else (160,160,160)
            cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            ytxt = max(30, y1 - 10)
            if live_prob is not None:
                cv.putText(frame, f"live={live_prob:.3f}", (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                ytxt += 24
            if yaw is not None:
                cv.putText(frame, f"yaw={yaw:+.3f}", (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                ytxt += 24
            if match_score is not None:
                cv.putText(frame, f"best={match_score:.3f}", (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                ytxt += 24
            cv.putText(frame, status, (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

        with lock:
            state["last_live_prob"] = live_prob
            state["last_yaw"] = yaw
            state["last_status"] = status
            state["last_badge"] = badge
            state["last_match"] = match_name
            state["last_match_score"] = match_score

        ok, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, threaded=True)
PY

clear
cd ~/project
source ~/fr-venv/bin/activate
python app.py
cat > ~/project/app.py <<'PY'
#!/usr/bin/env python3
from flask import Flask, Response, request, jsonify
import cv2 as cv
import numpy as np
import os, time, json, subprocess
from threading import Lock

APP_HOST = "0.0.0.0"
APP_PORT = 5000

BASE_DIR = "/home/pi/project"
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "db")
USERS_JSON = os.path.join(DB_DIR, "users.json")

MODEL_YUNET = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
MODEL_LIVE  = os.path.join(MODEL_DIR, "liveness.onnx")
MODEL_SFACE = os.path.join(MODEL_DIR, "face_recognition_sface.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# ---------- utils ----------
def download(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = subprocess.run(["curl", "-L", "-o", out_path, url],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {out_path}:\n{r.stderr}")

def validate_onnx_loadable(path: str):
    _ = cv.dnn.readNetFromONNX(path)

def ensure_model(path: str, url: str):
    if not os.path.exists(path):
        print("[INFO] downloading", path)
        download(url, path)
    try:
        validate_onnx_loadable(path)
        return
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        print("[INFO] re-downloading", path)
        download(url, path)
        validate_onnx_loadable(path)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def yaw_from_landmarks(lmk_full):
    # (x,y)*5: left_eye, right_eye, nose, mouth_l, mouth_r
    lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry = [float(v) for v in lmk_full[:10]]
    eye_mid_x = (lx + rx) * 0.5
    eye_dist = abs(rx - lx) + 1e-6
    return (nx - eye_mid_x) / eye_dist

def is_blurry(img, thresh=60.0):
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(g, cv.CV_64F).var() < thresh

def load_users():
    os.makedirs(DB_DIR, exist_ok=True)
    if not os.path.exists(USERS_JSON):
        return {}
    with open(USERS_JSON, "r") as f:
        return json.load(f)

def save_users(users):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(USERS_JSON, "w") as f:
        json.dump(users, f, indent=2)

def save_user_embedding(name: str, emb_list):
    embs = np.stack(emb_list).astype(np.float32)
    mean = embs.mean(axis=0)
    out = os.path.join(DB_DIR, f"{name}.npz")
    np.savez_compressed(out, embs=embs, mean=mean)

def load_user_embedding(name: str):
    path = os.path.join(DB_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["mean"].astype(np.float32)

# ---------- OpenCV tuning ----------
try:
    cv.setUseOptimized(True)
    cv.setNumThreads(0)
except Exception:
    pass

# ---------- Ensure models ----------
ensure_model(MODEL_YUNET, YUNET_URL)
ensure_model(MODEL_SFACE, SFACE_URL)
if not (os.path.exists(MODEL_LIVE) and os.path.getsize(MODEL_LIVE) > 100_000):
    raise RuntimeError(f"Missing liveness model: {MODEL_LIVE}")
print("[INFO] Models OK")

# ---------- Load models ----------
detector = cv.FaceDetectorYN.create(MODEL_YUNET, "", (320, 320), 0.6, 0.3, 5000)
recognizer = cv.FaceRecognizerSF.create(MODEL_SFACE, "")
live_net = cv.dnn.readNetFromONNX(MODEL_LIVE)

# ---------- Camera ----------
cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

# ---------- Calibrate liveness "real" index ----------
real_idx = None
calib_logits = []
t0 = time.time()
while time.time() - t0 < 1.5:
    ok, frame = cap.read()
    if not ok:
        continue

    h, w = frame.shape[:2]
    small = cv.resize(frame, (320, 180), interpolation=cv.INTER_LINEAR)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        continue

    # pick biggest for calibration
    f = max(faces, key=lambda x: x[2] * x[3])
    x, y, bw, bh = f[:4].astype(int)

    sx = w / small.shape[1]
    sy = h / small.shape[0]
    x1 = int(x * sx); y1 = int(y * sy)
    x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)

    pad = int(0.20 * max(x2 - x1, y2 - y1))
    x1 = clamp(x1 - pad, 0, w - 1); y1 = clamp(y1 - pad, 0, h - 1)
    x2 = clamp(x2 + pad, 0, w - 1); y2 = clamp(y2 + pad, 0, h - 1)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        continue

    blob = cv.dnn.blobFromImage(face, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    live_net.setInput(blob)
    out = live_net.forward()
    logits = out.reshape(-1).astype(np.float32)
    if logits.size >= 2:
        calib_logits.append(logits[:2])

if calib_logits:
    mean_logits = np.mean(np.stack(calib_logits), axis=0)
    real_idx = int(np.argmax(mean_logits))
print("[INFO] real_idx =", real_idx)

# ---------- Liveness + recognition helpers ----------
def liveness_score(face_bgr):
    blob = cv.dnn.blobFromImage(face_bgr, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    live_net.setInput(blob)
    out = live_net.forward()
    logits = out.reshape(-1).astype(np.float32)[:2]
    p = softmax(logits)
    return float(p[real_idx]) if real_idx is not None else float(np.max(p))

def face_embedding(frame, bbox, landmarks_full):
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    face_info = np.array([x1,y1,w,h, *landmarks_full.tolist()], dtype=np.float32)
    aligned = recognizer.alignCrop(frame, face_info)
    feat = recognizer.feature(aligned)  # (1, 128)
    return feat.flatten().astype(np.float32)

def get_faces(frame):
    """
    Returns list of faces sorted by area desc:
    each item: (area, bbox(x1,y1,x2,y2), landmarks_full(10 floats))
    """
    h, w = frame.shape[:2]
    small = cv.resize(frame, (320, 180), interpolation=cv.INTER_LINEAR)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        return []

    sx = w / small.shape[1]
    sy = h / small.shape[0]
    out = []

    for f in faces:
        x, y, bw, bh = f[:4].astype(int)
        lmk = f[4:14]

        x1 = int(x * sx); y1 = int(y * sy)
        x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)

        pad = int(0.20 * max(x2 - x1, y2 - y1))
        x1 = clamp(x1 - pad, 0, w - 1); y1 = clamp(y1 - pad, 0, h - 1)
        x2 = clamp(x2 + pad, 0, w - 1); y2 = clamp(y2 + pad, 0, h - 1)

        lmk_full = []
        for i in range(0, 10, 2):
            lmk_full.append(float(lmk[i]) * sx)
            lmk_full.append(float(lmk[i+1]) * sy)

        area = (x2 - x1) * (y2 - y1)
        out.append((area, (x1,y1,x2,y2), np.array(lmk_full, dtype=np.float32)))

    out.sort(key=lambda t: t[0], reverse=True)
    return out

# ---------- Shared state for UI ----------
lock = Lock()
state = {
    "enrolling": False,
    "enroll_name": None,
    "enroll_count": 0,
    "enroll_target": 30,
    "last_status": "idle",
    "last_badge": "gray",   # green/yellow/red/gray
    "known_users": [],
    "faces": [],            # list of per-face status objects
}

# ---------- Web UI ----------
HOME_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pi Face System</title>
<style>
  body { font-family: system-ui, Arial; margin: 20px; }
  .row { display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }
  .card { border:1px solid #ddd; border-radius:12px; padding:16px; }
  .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; color:#fff; }
  .green { background:#1f9d55; }
  .yellow{ background:#d97706; }
  .red   { background:#dc2626; }
  .gray  { background:#6b7280; }
  .btn { display:inline-block; padding:10px 14px; border-radius:10px; border:1px solid #ddd; background:#f8fafc; cursor:pointer; text-decoration:none; color:#111; }
  .btn:hover { background:#eef2f7; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  table { border-collapse: collapse; width:100%; }
  td, th { border-bottom:1px solid #eee; padding:6px 8px; text-align:left; font-size:14px; }
</style>
</head>
<body>
  <h2>Pi Face Recognition</h2>

  <div class="row">
    <div class="card" style="min-width:360px;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div id="badge" class="badge gray">IDLE</div>
        <div class="mono" id="statusText">loading...</div>
      </div>

      <div style="margin-top:10px">
        <div>Users: <span class="mono" id="users">-</span></div>
      </div>

      <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
        <a class="btn" href="/enroll">Go to Enrollment</a>
        <button class="btn" onclick="refreshNow()">Refresh</button>
      </div>

      <h3 style="margin-top:16px;">Faces</h3>
      <table>
        <thead><tr><th>#</th><th>Status</th><th>Live</th><th>Yaw</th><th>Match</th></tr></thead>
        <tbody id="facesTbody"></tbody>
      </table>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Live Feed</h3>
      <img src="/video" style="max-width:900px; width:100%; border-radius:10px;"/>
      <div style="margin-top:8px; color:#555; font-size:14px;">
        Multi-face is supported. Enrollment is only allowed when exactly one face is visible.
      </div>
    </div>
  </div>

<script>
function badgeToClass(b){ return (b==='green'||b==='yellow'||b==='red') ? b : 'gray'; }

async function refreshNow(){
  const r = await fetch('/api/status');
  const s = await r.json();

  const badge = document.getElementById('badge');
  badge.className = 'badge ' + badgeToClass(s.last_badge);
  badge.textContent = (s.last_badge || 'gray').toUpperCase();

  document.getElementById('statusText').textContent = s.last_status || '-';
  document.getElementById('users').textContent = (s.known_users && s.known_users.length) ? s.known_users.join(', ') : '-';

  const tb = document.getElementById('facesTbody');
  tb.innerHTML = '';
  const faces = s.faces || [];
  faces.forEach((f, idx) => {
    const tr = document.createElement('tr');
    const live = (f.live_prob==null) ? '-' : f.live_prob.toFixed(3);
    const yaw  = (f.yaw==null) ? '-' : f.yaw.toFixed(3);
    const match = f.match_name ? (f.match_name + (f.match_score!=null ? ` (${f.match_score.toFixed(3)})` : '')) : '-';
    tr.innerHTML = `<td>${idx+1}</td><td>${f.status||'-'}</td><td>${live}</td><td>${yaw}</td><td>${match}</td>`;
    tb.appendChild(tr);
  });
}
setInterval(refreshNow, 500);
refreshNow();
</script>
</body>
</html>
"""

ENROLL_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Enroll</title>
<style>
  body { font-family: system-ui, Arial; margin: 20px; }
  .row { display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }
  .card { border:1px solid #ddd; border-radius:12px; padding:16px; }
  .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; color:#fff; }
  .green { background:#1f9d55; }
  .yellow{ background:#d97706; }
  .red   { background:#dc2626; }
  .gray  { background:#6b7280; }
  .btn { padding:10px 14px; border-radius:10px; border:1px solid #ddd; background:#f8fafc; cursor:pointer; }
  .btn:hover { background:#eef2f7; }
  input { padding:10px; border-radius:10px; border:1px solid #ddd; width:260px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .bar { width: 100%; height: 16px; background:#eee; border-radius: 999px; overflow:hidden; }
  .fill { height: 100%; width: 0%; background:#1f9d55; transition: width 0.2s ease; }
  .hint { color:#555; font-size:14px; margin-top:10px; }
</style>
</head>
<body>
  <h2>Enrollment</h2>
  <div style="margin-bottom:12px;">
    <a href="/" class="btn" style="text-decoration:none; color:#111;">← Back</a>
  </div>

  <div class="row">
    <div class="card" style="min-width:360px;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div id="badge" class="badge gray">IDLE</div>
        <div class="mono" id="statusText">loading...</div>
      </div>

      <div style="margin-top:10px;">
        <div>Enrolling: <span class="mono" id="enrolling">-</span></div>
        <div>Progress: <span class="mono" id="progressText">0/0</span></div>
        <div class="bar" style="margin-top:8px;"><div id="fill" class="fill"></div></div>
      </div>

      <hr style="margin:16px 0; border:none; border-top:1px solid #eee;">

      <form id="enrollForm">
        <label>Name</label><br>
        <input id="name" placeholder="e.g. alice" required>
        <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
          <button class="btn" type="submit">Start Enrollment</button>
          <button class="btn" type="button" onclick="stopEnroll()">Stop</button>
        </div>
      </form>

      <div class="hint">
        Enrollment rule: <b>exactly one face</b> must be visible.
        If another face appears, enrollment will cancel and you must restart.
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Live Feed</h3>
      <img src="/video" style="max-width:900px; width:100%; border-radius:10px;"/>
    </div>
  </div>

<script>
async function api(url, opts){
  const r = await fetch(url, opts);
  const j = await r.json();
  if(!r.ok) throw new Error(j.error || 'request failed');
  return j;
}

document.getElementById('enrollForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = document.getElementById('name').value.trim();
  if(!name) return;

  try{
    await api('/api/enroll/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({name})
    });
  }catch(err){
    alert(err.message);
  }
});

async function stopEnroll(){
  try{ await api('/api/enroll/stop', {method:'POST'}); }
  catch(err){ alert(err.message); }
}

function badgeToClass(b){ return (b==='green'||b==='yellow'||b==='red') ? b : 'gray'; }

async function refresh(){
  const s = await (await fetch('/api/status')).json();

  const badge = document.getElementById('badge');
  badge.className = 'badge ' + badgeToClass(s.last_badge);
  badge.textContent = (s.last_badge || 'gray').toUpperCase();

  document.getElementById('statusText').textContent = s.last_status || '-';
  document.getElementById('enrolling').textContent = s.enrolling ? (s.enroll_name || '-') : 'no';

  const c = s.enroll_count || 0;
  const t = s.enroll_target || 0;
  document.getElementById('progressText').textContent = `${c}/${t}`;
  const pct = t ? Math.min(100, Math.round((c / t) * 100)) : 0;
  const fill = document.getElementById('fill');
  fill.style.width = pct + '%';
  fill.style.background = (s.last_badge === 'green') ? '#1f9d55'
                    : (s.last_badge === 'yellow') ? '#d97706'
                    : (s.last_badge === 'red') ? '#dc2626'
                    : '#6b7280';
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/")
def home():
    return HOME_HTML

@app.route("/enroll")
def enroll_page():
    return ENROLL_HTML

@app.route("/api/status")
def api_status():
    users = load_users()
    with lock:
        state["known_users"] = sorted(list(users.keys()))
        return jsonify(state)

@app.route("/api/enroll/start", methods=["POST"])
def api_enroll_start():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name is required"}), 400

    with lock:
        state["enrolling"] = True
        state["enroll_name"] = name
        state["enroll_count"] = 0
        state["enroll_target"] = 30
        state["last_status"] = f"enrolling {name} (show ONLY one face)"
        state["last_badge"] = "yellow"
    return jsonify({"ok": True, "name": name})

@app.route("/api/enroll/stop", methods=["POST"])
def api_enroll_stop():
    with lock:
        state["enrolling"] = False
        state["enroll_name"] = None
        state["enroll_count"] = 0
        state["last_status"] = "enrollment stopped"
        state["last_badge"] = "gray"
    return jsonify({"ok": True})

# ---------- Video processing loop ----------
def gen_frames():
    # performance caps
    MAX_FACES = 3

    # thresholds/tuning
    PASSIVE_THRESH = 0.92
    LIVE_HITS_REQUIRED = 5
    RECOG_THRESH = 0.35
    ENROLL_TARGET = 30

    # per-slot state (slot = index in sorted-by-area list)
    hits = [0] * MAX_FACES
    last_left_time = [0.0] * MAX_FACES
    last_right_time = [0.0] * MAX_FACES

    CHALLENGE_WINDOW = 4.0
    YAW_LEFT  = -0.18
    YAW_RIGHT = +0.18

    # enrollment samples (only for single-face mode)
    enroll_samples = []

    # load known embeddings
    users = load_users()
    known = {name: load_user_embedding(name) for name in users.keys()}

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        now = time.time()

        faces = get_faces(frame)
        faces = faces[:MAX_FACES]

        # --- Enrollment rule enforcement ---
        with lock:
            enrolling = state["enrolling"]
            enroll_name = state["enroll_name"]

        if enrolling and len(faces) != 1:
            # Cancel enrollment and force restart
            enroll_samples = []
            with lock:
                state["enrolling"] = False
                state["enroll_name"] = None
                state["enroll_count"] = 0
                state["last_status"] = "ENROLL CANCELLED: multiple faces detected. Please restart enrollment."
                state["last_badge"] = "red"

        face_statuses = []

        # default overall status
        overall_status = "NO FACE"
        overall_badge = "gray"

        # process faces
        for i, (area, bbox, lmk_full) in enumerate(faces):
            x1,y1,x2,y2 = bbox
            face_img = frame[y1:y2, x1:x2]

            st = {
                "slot": i,
                "status": "FACE",
                "badge": "gray",
                "live_prob": None,
                "yaw": None,
                "match_name": None,
                "match_score": None,
            }

            if face_img.size == 0 or (x2-x1) < 160 or is_blurry(face_img, 60.0):
                st["status"] = "LOW QUALITY"
                st["badge"] = "yellow"
                hits[i] = max(0, hits[i] - 1)
            else:
                live_prob = liveness_score(face_img)
                yaw = yaw_from_landmarks(lmk_full)

                st["live_prob"] = live_prob
                st["yaw"] = yaw

                # active challenge per slot
                if yaw <= YAW_LEFT:
                    last_left_time[i] = now
                if yaw >= YAW_RIGHT:
                    last_right_time[i] = now

                left_ok = (now - last_left_time[i]) <= CHALLENGE_WINDOW
                right_ok = (now - last_right_time[i]) <= CHALLENGE_WINDOW
                active_ok = left_ok and right_ok
                passive_ok = (live_prob >= PASSIVE_THRESH)

                if passive_ok and active_ok:
                    hits[i] += 1
                else:
                    hits[i] = max(0, hits[i] - 1)

                live_ok = hits[i] >= LIVE_HITS_REQUIRED

                if not passive_ok:
                    st["status"] = "SPOOF? (PASSIVE FAIL)"
                    st["badge"] = "red"
                elif not active_ok:
                    st["status"] = "TURN HEAD L/R"
                    st["badge"] = "yellow"
                elif not live_ok:
                    st["status"] = "HOLD..."
                    st["badge"] = "yellow"
                else:
                    st["status"] = "LIVE OK"
                    st["badge"] = "green"

                # recognition only if live_ok
                if live_ok:
                    emb = face_embedding(frame, bbox, lmk_full)

                    best_name = None
                    best_sim = -1.0
                    for name, ref in known.items():
                        if ref is None:
                            continue
                        sim = cosine_sim(emb, ref)
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name

                    if best_name is not None and best_sim >= RECOG_THRESH:
                        st["match_name"] = best_name
                        st["match_score"] = best_sim
                        st["status"] = f"VERIFIED: {best_name}"
                        st["badge"] = "green"
                    else:
                        st["match_score"] = best_sim if best_sim > -1 else None
                        st["status"] = "UNKNOWN"
                        st["badge"] = "yellow"

                    # Enrollment capture ONLY when exactly one face exists and enrollment is active
                    with lock:
                        enrolling_now = state["enrolling"]
                        enroll_name_now = state["enroll_name"]

                    if enrolling_now and enroll_name_now and len(faces) == 1 and i == 0:
                        enroll_samples.append(emb)
                        with lock:
                            state["enroll_target"] = ENROLL_TARGET
                            state["enroll_count"] = len(enroll_samples)
                            state["last_status"] = f"enrolling {enroll_name_now} ({len(enroll_samples)}/{ENROLL_TARGET})"
                            state["last_badge"] = "yellow"

                        if len(enroll_samples) >= ENROLL_TARGET:
                            save_user_embedding(enroll_name_now, enroll_samples)
                            users = load_users()
                            users[enroll_name_now] = {"created": time.time(), "count": len(enroll_samples)}
                            save_users(users)
                            known[enroll_name_now] = load_user_embedding(enroll_name_now)
                            enroll_samples = []

                            with lock:
                                state["enrolling"] = False
                                state["enroll_name"] = None
                                state["last_status"] = f"enrolled {enroll_name_now}"
                                state["last_badge"] = "green"

            # draw bbox + labels
            if st["badge"] == "green":
                color = (0,255,0)
            elif st["badge"] == "yellow":
                color = (0,255,255)
            elif st["badge"] == "red":
                color = (0,0,255)
            else:
                color = (160,160,160)

            cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            ytxt = max(30, y1 - 10)

            if st["live_prob"] is not None:
                cv.putText(frame, f"live={st['live_prob']:.3f}", (x1, ytxt),
                           cv.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                ytxt += 22
            if st["yaw"] is not None:
                cv.putText(frame, f"yaw={st['yaw']:+.3f}", (x1, ytxt),
                           cv.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                ytxt += 22
            if st["match_name"] is not None and st["match_score"] is not None:
                cv.putText(frame, f"{st['match_name']} {st['match_score']:.3f}", (x1, ytxt),
                           cv.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                ytxt += 22
            cv.putText(frame, st["status"], (x1, ytxt),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

            face_statuses.append(st)

        # overall badge/status for UI
        if len(faces) == 0:
            overall_status = "NO FACE"
            overall_badge = "gray"
        else:
            # priority: red > green(verified/live) > yellow > gray
            badges = [fs["badge"] for fs in face_statuses]
            statuses = [fs["status"] for fs in face_statuses]

            if any("VERIFIED:" in s for s in statuses):
                overall_status = "VERIFIED"
                overall_badge = "green"
            elif "red" in badges:
                overall_status = "SPOOF?/FAIL"
                overall_badge = "red"
            elif "yellow" in badges:
                overall_status = "CHECKING"
                overall_badge = "yellow"
            else:
                overall_status = "LIVE OK"
                overall_badge = "green"

        # update shared state for UI
        with lock:
            state["faces"] = face_statuses
            # only override last_* if not currently showing explicit enrollment cancelled message
            if not (state["last_status"].startswith("ENROLL CANCELLED")):
                state["last_status"] = overall_status
                state["last_badge"] = overall_badge

        # MJPEG encode
        ok, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, threaded=True)
PY

cd ~/project
source ~/fr-venv/bin/activate
python app.py
clear
shutdown 0
sudo shutdown 0
clear
sudo nano /etc/dhcpcd.conf
sudo reboot
ip -br a
ip link show eth0
sudo ethtool eth0 2>/dev/null | egrep -i "Link detected|Speed|Duplex" || echo "ethtool not installed"
clear
grep -n "interface eth0" -A8 /etc/dhcpcd.conf
tail -n 30 /etc/dhcpcd.conf
clear
systemctl is-active dhcpcd
nmcli dev status 2>/dev/null || echo "NetworkManager not installed"
sudo nmcli con add type ethernet ifname eth0 con-name pi-direct-eth ipv4.method manual ipv4.addresses 192.168.50.2/24 ipv4.gateway "" ipv4.dns "" ipv6.method ignore autoconnect yes
sudo nmcli con up pi-direct-eth
ip -br a
nmcli dev status
ping -c 3 192.168.50.1
clear
sudo apt update
sudo apt install -y speedtest-cli
speedtest-cli
ip route get 1.1.1.1\
ip route get 1.1.1.1
iperf3 -s
clear
nmcli dev status
iw dev wlan0 link
ip route
sudo nmcli radio wifi off
ip route get 1.1.1.1
sudo nmcli con modify pi-direct-eth ipv4.route-metric 100
sudo nmcli con modify netplan-wlan0-1 ipv4.route-metric 600
sudo nmcli con down pi-direct-eth && sudo nmcli con up pi-direct-eth
sudo nmcli con down netplan-wlan0-1 && sudo nmcli con up netplan-wlan0-1
sudo apt upadte
sudo apt upadate
sudo apt-get update
sudo apt update
sudo apt install -y speedtest-cli
speedtest-cli
sudo apt install -y iperf3
iperf3 -c 192.168.137.1
sudo apt update
clear
sudo nmcli con modify pi-direct-eth ipv4.addresses 192.168.137.2/24 ipv4.gateway 192.168.137.1 ipv4.dns "192.168.137.1 1.1.1.1" ipv4.method manual
sudo nmcli con down pi-direct-eth
sudo nmcli con up pi-direct-eth
ip -br a
ping -c 3 192.168.137.1
ping -c 3 1.1.1.1
ping -c 3 google.com
sudo nmcli con modify netplan-wlan0-1 ipv4.route-metric 100
sudo nmcli con modify pi-direct-eth ipv4.route-metric 600
sudo nmcli con down netplan-wlan0-1 && sudo nmcli con up netplan-wlan0-1
sudo nmcli con down pi-direct-eth && sudo nmcli con up pi-direct-eth
ssh pi@192.168.137.2
cls
clear
cd test/
cd ..
ls
cd project/
ls
python app.py 
cd project/
sudo apt update
cd project/
python3 app.py 
sudo reboot
clear
cd project/
python3 app.py 
cd project/
python3 app.py 
sudo ethtool -s eth0 speed 100 duplex full autoneg off
sudo nmcli con modify pi-direct-eth ethernet.auto-negotiate no ethernet.speed 100 ethernet.duplex full
sudo nmcli con down pi-direct-eth && sudo nmcli con up pi-direct-eth
ping -t 192.168.137.2
sudo ethtool eth0 | egrep -i "Speed|Duplex|Auto-negotiation|Link detected"
sudo ethtool -s eth0 autoneg on
sudo nmcli con modify netplan-wlan0-1 ipv4.route-metric "" ipv6.route-metric ""
sudo nmcli con down netplan-wlan0-1 && sudo nmcli con up netplan-wlan0-1
sudo nmcli radio wifi on
ip -br a
ip route
sudo reboot
cd project/
python3 app.py 
sudo shutdown 0
clear
cd project/
python a
python app.py 
cd project/
python app.py 
clear
cat app.py 
clear
msedit app.py 
ls
msedit project.py
pinout
python3 project.py 
ls
ls /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
sudo ls /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
ls /usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml
tree
python3 -c "import cv2; print(cv2.__version__)"
ls /usr/share/opencv4/haarcascades/ | head
cat app.py 
clear
ls /static
cd static/
ls
ls -a
ls -l
cd ..
clear
sudo apt update
sudo apt install -y python3-flask python3-opencv python3-gpiozero python3-lgpio
msedit project.py 
clear
python3 project.py 
ls
cd ..
ls
cd test/
ls
cat servo_toggle_0_90.py 
clear
cd ..
cd project/
msedit project.py 
python project.py 
cd ..
cd test/
python servo_toggle_0_90.py 
cat servo_toggle_0_90.py 
cd ..
cd project/
msedit door_controller.py
msedit project.py 
cls
clear
msedit project.py 
python project.py 
cat project.py 
msedit project.py 
clear
python project.py 
sudo shutdown 0
ls
ls project/
ls test
ls
git init
