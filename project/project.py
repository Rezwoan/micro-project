#!/usr/bin/env python3
from flask import Flask, Response, request, jsonify, render_template_string, send_from_directory
import cv2 as cv
import numpy as np
import os, time, threading, atexit

from gpiozero import LED, Button, Buzzer
from door_controller import DoorController

# =========================
# CONFIG
# =========================
APP_HOST = "0.0.0.0"
APP_PORT = 5000

# GPIO (BCM)
PIN_LED = 21          # Pin 40
PIN_BUZZER = 23       # Pin 16
PIN_BTN_OUTSIDE = 17  # Pin 11 (doorbell button outside)
PIN_BTN_INSIDE  = 27  # Pin 13 (entered/close button inside)

# Door controller (your module)
DOOR_GPIO = 18

# Buzzer module trigger mode (many active modules are active HIGH)
BUZZER_ACTIVE_HIGH = True  # set False if your buzzer triggers on LOW

# Face thresholds (tune if needed)
RECOG_THRESH = 0.35
LIVE_THRESH  = 0.85

# Auto open/close behavior
FACE_COOLDOWN_SEC = 5.0
AUTO_CLOSE_SEC = 25  # safety close after open

# Camera
CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720
PROC_W, PROC_H = 320, 180
STREAM_JPEG_QUALITY = 80

# Paths (your existing structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
USERS_JSON = os.path.join(DB_DIR, "users.json")

MODEL_YUNET = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
MODEL_LIVE  = os.path.join(MODEL_DIR, "liveness.onnx")
MODEL_SFACE = os.path.join(MODEL_DIR, "face_recognition_sface.onnx")

# Optional: only allow specific enrolled names to auto-open
# AUTHORIZED_USERS = {"panda"}  # example
AUTHORIZED_USERS = set()

os.makedirs(STATIC_DIR, exist_ok=True)

# =========================
# Hardware
# =========================
led = LED(PIN_LED, initial_value=False)
buzzer = Buzzer(PIN_BUZZER, active_high=BUZZER_ACTIVE_HIGH, initial_value=False)

btn_outside = Button(PIN_BTN_OUTSIDE, pull_up=True, bounce_time=0.05)
btn_inside  = Button(PIN_BTN_INSIDE,  pull_up=True, bounce_time=0.05)

door = DoorController(gpio_pin=DOOR_GPIO)

door_lock = threading.Lock()
last_open_ts = 0.0

# =========================
# Cleanup
# =========================
cap = None

def cleanup_all():
    try:
        buzzer.off()
    except Exception:
        pass
    try:
        led.off()
    except Exception:
        pass
    try:
        door.cleanup()
    except Exception:
        pass
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

atexit.register(cleanup_all)

# =========================
# Models (YuNet + SFace + Liveness)
# =========================
for p in (MODEL_YUNET, MODEL_SFACE, MODEL_LIVE):
    if not os.path.exists(p):
        raise RuntimeError(f"Missing model: {p}")

try:
    cv.setUseOptimized(True)
    cv.setNumThreads(0)
except Exception:
    pass

detector = cv.FaceDetectorYN.create(MODEL_YUNET, "", (PROC_W, PROC_H), 0.6, 0.3, 5000)
recognizer = cv.FaceRecognizerSF.create(MODEL_SFACE, "")
live_net = cv.dnn.readNetFromONNX(MODEL_LIVE)

# =========================
# DB helpers (same style as your app.py)
# =========================
def load_users():
    if not os.path.exists(USERS_JSON):
        return {}
    import json
    with open(USERS_JSON, "r") as f:
        return json.load(f)

def load_user_embedding(name: str):
    path = os.path.join(DB_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["mean"].astype(np.float32)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

known_lock = threading.Lock()
known = {}

def reload_known():
    users = load_users()
    tmp = {}
    for name in users.keys():
        emb = load_user_embedding(name)
        if emb is not None:
            tmp[name] = emb
    with known_lock:
        known.clear()
        known.update(tmp)
    return sorted(list(tmp.keys()))

reload_known()

# =========================
# Camera
# =========================
cap = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv.CAP_PROP_FPS, 30)

frame_lock = threading.Lock()
latest_frame = None
latest_jpeg = None

def save_snapshot(path):
    with frame_lock:
        f = None if latest_frame is None else latest_frame.copy()
    if f is None:
        return False
    return bool(cv.imwrite(path, f, [int(cv.IMWRITE_JPEG_QUALITY), 92]))

# =========================
# Shared State for UI
# =========================
state_lock = threading.Lock()
state = {
    "door_open": False,
    "led_on": False,
    "face_auto_enabled": True,
    "last_verified": None,
    "last_score": None,
    "last_live": None,
    "users": [],
    "doorbell_pending": False,
    "doorbell_time": None,
    "doorbell_image": None,
    "error": None,
    "last_event": None,
}

def set_error(msg):
    with state_lock:
        state["error"] = msg

def set_event(msg):
    with state_lock:
        state["last_event"] = msg

def led_set(on: bool):
    if on:
        led.on()
    else:
        led.off()
    with state_lock:
        state["led_on"] = on

def ring_bell():
    def _w():
        try:
            for _ in range(2):
                buzzer.on(); time.sleep(0.12)
                buzzer.off(); time.sleep(0.08)
            time.sleep(0.12)
            for _ in range(2):
                buzzer.on(); time.sleep(0.12)
                buzzer.off(); time.sleep(0.08)
        except Exception:
            pass
    threading.Thread(target=_w, daemon=True).start()

def door_open(reason="manual"):
    global last_open_ts
    with door_lock:
        door.open_door()
        last_open_ts = time.time()
        led_set(True)
        with state_lock:
            state["door_open"] = True
        set_event(f"Door opened ({reason})")

def door_close(reason="manual"):
    with door_lock:
        door.close_door()
        led_set(False)
        with state_lock:
            state["door_open"] = False
        set_event(f"Door closed ({reason})")

def inside_entered():
    door_close(reason="inside_entered")

def outside_doorbell():
    try:
        img_path = os.path.join(STATIC_DIR, "doorbell.jpg")
        ok = save_snapshot(img_path)
        ring_bell()
        with state_lock:
            state["doorbell_pending"] = True
            state["doorbell_time"] = int(time.time())
            state["doorbell_image"] = "/static/doorbell.jpg" if ok else None
        set_event("Doorbell pressed")
    except Exception as e:
        set_error(f"doorbell: {e}")

btn_inside.when_pressed = inside_entered
btn_outside.when_pressed = outside_doorbell

# =========================
# Face pipeline (YuNet + SFace + Liveness)
# =========================
def detect_faces(frame):
    h0, w0 = frame.shape[:2]
    small = cv.resize(frame, (PROC_W, PROC_H), interpolation=cv.INTER_LINEAR)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        return []

    sx = w0 / small.shape[1]
    sy = h0 / small.shape[0]
    out = []
    for f in faces:
        x, y, bw, bh = f[:4].astype(int)
        lmk = f[4:14]

        x1 = int(x * sx); y1 = int(y * sy)
        x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)

        lmk_full = []
        for i in range(0, 10, 2):
            lmk_full.append(float(lmk[i]) * sx)
            lmk_full.append(float(lmk[i+1]) * sy)
        lmk_full = np.array(lmk_full, dtype=np.float32)

        area = (x2 - x1) * (y2 - y1)
        out.append((area, (x1, y1, x2, y2), lmk_full))

    out.sort(key=lambda t: t[0], reverse=True)
    return out

real_idx = None
calib_logits = []

def liveness_prob(face_bgr):
    blob = cv.dnn.blobFromImage(face_bgr, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    live_net.setInput(blob)
    out = live_net.forward().reshape(-1).astype(np.float32)[:2]
    p = softmax(out)
    if real_idx is None:
        return float(np.max(p))
    return float(p[real_idx])

def face_embedding(frame, bbox, landmarks_full):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    face_info = np.array([x1, y1, w, h, *landmarks_full.tolist()], dtype=np.float32)
    aligned = recognizer.alignCrop(frame, face_info)
    feat = recognizer.feature(aligned)
    return feat.flatten().astype(np.float32)

# =========================
# Processor thread
# =========================
last_face_open = 0.0

def processor():
    global latest_frame, latest_jpeg, real_idx, calib_logits, last_face_open

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        with frame_lock:
            latest_frame = frame

        # calibrate "real" index quickly (like your app.py)
        if real_idx is None and len(calib_logits) < 20:
            fs = detect_faces(frame)
            if fs:
                _, bbox, _ = fs[0]
                x1,y1,x2,y2 = bbox
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    blob = cv.dnn.blobFromImage(face_img, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
                    live_net.setInput(blob)
                    out = live_net.forward().reshape(-1).astype(np.float32)
                    if out.size >= 2:
                        calib_logits.append(out[:2])
                        if len(calib_logits) >= 10:
                            mean_logits = np.mean(np.stack(calib_logits), axis=0)
                            real_idx = int(np.argmax(mean_logits))

        verified_name = None
        verified_score = None
        live_p = None

        fs = detect_faces(frame)
        if fs:
            _, bbox, lmk_full = fs[0]  # biggest face
            x1,y1,x2,y2 = bbox
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                live_p = liveness_prob(face_img)
                emb = face_embedding(frame, bbox, lmk_full)

                best_name = None
                best_sim = -1.0
                with known_lock:
                    items = list(known.items())

                for name, ref in items:
                    sim = cosine_sim(emb, ref)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

                allowed = (best_name is not None and best_sim >= RECOG_THRESH and live_p >= LIVE_THRESH)
                if AUTHORIZED_USERS:
                    allowed = allowed and (best_name in AUTHORIZED_USERS)

                if allowed:
                    verified_name = best_name
                    verified_score = best_sim
                    cv.putText(frame, f"VERIFIED: {best_name} sim={best_sim:.3f} live={live_p:.3f}",
                               (x1, max(30, y1-10)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    cv.putText(frame, f"UNKNOWN sim={best_sim:.3f} live={live_p:.3f}",
                               (x1, max(30, y1-10)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        with state_lock:
            state["users"] = sorted(list(known.keys()))
            state["last_live"] = None if live_p is None else float(live_p)
            if verified_name:
                state["last_verified"] = verified_name
                state["last_score"] = float(verified_score)

        # auto-open
        with state_lock:
            auto = state["face_auto_enabled"]
            is_open = state["door_open"]

        now = time.time()
        if auto and (not is_open) and verified_name:
            if (now - last_face_open) > FACE_COOLDOWN_SEC:
                try:
                    door_open(reason=f"face:{verified_name}")
                    last_face_open = now
                except Exception as e:
                    set_error(f"auto-open: {e}")

        # auto-close safety
        with state_lock:
            is_open = state["door_open"]
        if is_open and (time.time() - last_open_ts) > AUTO_CLOSE_SEC:
            try:
                door_close(reason="auto_close")
            except Exception as e:
                set_error(f"auto-close: {e}")

        ok2, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
        if ok2:
            with frame_lock:
                latest_jpeg = buf.tobytes()

        time.sleep(0.01)

threading.Thread(target=processor, daemon=True).start()

# =========================
# Web UI
# =========================
HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Smart Door</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --bg:#0b1020; --panel:#111a2f;
    --text:#e8eefc; --muted:#a9b6d6;
    --good:#2ee59d; --warn:#ffb020; --bad:#ff4d5a;
    --stroke:rgba(255,255,255,.10); --radius:16px;
    --shadow: 0 10px 30px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{
    margin:0; padding:16px; color:var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
    background: radial-gradient(900px 600px at 20% 0%, rgba(79,140,255,.22), transparent 60%),
                radial-gradient(700px 500px at 100% 10%, rgba(46,229,157,.12), transparent 55%),
                var(--bg);
    display:flex; justify-content:center;
  }
  .wrap{width:100%; max-width:1100px}
  .top{display:flex; justify-content:space-between; align-items:flex-end; gap:10px; margin-bottom:14px}
  .title{font-size:20px; font-weight:800}
  .sub{color:var(--muted); font-size:13px; margin-top:6px}
  .pill{padding:10px 12px; border-radius:999px; border:1px solid var(--stroke); background:rgba(255,255,255,.06); color:var(--muted); font-size:12px}
  .grid{display:grid; grid-template-columns: 1.2fr .8fr; gap:14px}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
  .card{border:1px solid var(--stroke); background:linear-gradient(180deg, rgba(17,26,47,.92), rgba(15,23,42,.90)); border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden}
  .hd{display:flex; justify-content:space-between; align-items:center; padding:14px; border-bottom:1px solid rgba(255,255,255,.06)}
  .hd h3{margin:0; font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
  .badge{font-size:12px; padding:8px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); color:var(--muted)}
  .body{padding:14px}
  .video{width:100%; aspect-ratio:16/9; background:#000; border-radius:14px; overflow:hidden; border:1px solid rgba(255,255,255,.12)}
  .video img{width:100%; height:100%; object-fit:cover; display:block}
  .kvs{display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:12px}
  .kv{padding:12px; border-radius:14px; border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.04)}
  .kv .k{color:var(--muted); font-size:12px}
  .kv .v{margin-top:8px; font-weight:800}
  .v.good{color:var(--good)} .v.warn{color:var(--warn)} .v.bad{color:var(--bad)}
  .row{display:flex; gap:10px; flex-wrap:wrap}
  .btn{
    border:1px solid rgba(79,140,255,.35); background:rgba(79,140,255,.16);
    color:var(--text); font-weight:800; padding:12px 12px; border-radius:12px;
    cursor:pointer; transition:.12s ease; min-width:150px
  }
  .btn:hover{transform:translateY(-1px); filter:brightness(1.05)}
  .btn.good{border-color:rgba(46,229,157,.35); background:rgba(46,229,157,.14)}
  .btn.bad{border-color:rgba(255,77,90,.35); background:rgba(255,77,90,.14)}
  .btn.warn{border-color:rgba(255,176,32,.30); background:rgba(255,176,32,.12)}
  .note{color:var(--muted); font-size:12px; margin-top:10px; line-height:1.4}
  .snap img{width:100%; border-radius:14px; border:1px solid rgba(255,255,255,.12); margin-top:10px; display:block}
  .toast{margin-top:12px; padding:12px; border-radius:14px; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.05); color:var(--muted); font-size:12px; white-space:pre-wrap}
  .switch{width:44px; height:26px; border-radius:999px; border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.06); position:relative; cursor:pointer}
  .knob{width:20px; height:20px; border-radius:999px; background:rgba(255,255,255,.85); position:absolute; top:2px; left:2px; transition:.15s ease}
  .switch.on{background:rgba(46,229,157,.16); border-color:rgba(46,229,157,.35)}
  .switch.on .knob{left:22px; background:rgba(46,229,157,.95)}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="title">Smart Door Dashboard</div>
      <div class="sub">Verified face → opens door • Outside button → doorbell + photo</div>
    </div>
    <div class="pill" id="pill">Loading…</div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="hd"><h3>Live Camera</h3><div class="badge">LIVE</div></div>
      <div class="body">
        <div class="video"><img src="/video_feed" alt="Live stream"></div>

        <div class="kvs">
          <div class="kv"><div class="k">Door</div><div class="v" id="doorV">—</div></div>
          <div class="kv"><div class="k">LED</div><div class="v" id="ledV">—</div></div>
          <div class="kv"><div class="k">Last Verified</div><div class="v" id="whoV">—</div></div>
          <div class="kv"><div class="k">Doorbell</div><div class="v" id="bellV">—</div></div>
        </div>

        <div class="note">
          Users list comes from <b>db/</b>. If empty, enroll with your enrollment script first.
        </div>
      </div>
    </div>

    <div class="card">
      <div class="hd"><h3>Controls</h3><div class="badge">Actions</div></div>
      <div class="body">
        <div class="row">
          <button class="btn good" onclick="act('open')">Open Door</button>
          <button class="btn bad"  onclick="act('close')">Close Door</button>
          <button class="btn"      onclick="act('entered')">I Entered (Close)</button>
          <button class="btn"      onclick="act('led_on')">LED On</button>
          <button class="btn"      onclick="act('led_off')">LED Off</button>
          <button class="btn warn" onclick="act('ring')">Ring Bell</button>
          <button class="btn"      onclick="act('reload')">Reload Users</button>
        </div>

        <div class="note" style="display:flex; align-items:center; gap:10px;">
          Face auto-open
          <div class="switch" id="faceSw" onclick="toggleFace()"><div class="knob"></div></div>
        </div>

        <div class="card" style="margin-top:14px;">
          <div class="hd"><h3>Doorbell Event</h3><div class="badge" id="bellBadge">—</div></div>
          <div class="body snap">
            <div class="note" id="bellNote">No event.</div>
            <div id="snapWrap"></div>
            <div class="row" style="margin-top:10px;">
              <button class="btn good" onclick="act('open')">Open Remotely</button>
              <button class="btn" onclick="act('clear_bell')">Clear</button>
              <button class="btn" onclick="act('snapshot')">Take Snapshot</button>
            </div>
          </div>
        </div>

        <div class="toast" id="errBox" style="display:none;"></div>
      </div>
    </div>
  </div>
</div>

<script>
async function getState(){ return await (await fetch("/api/state")).json(); }

async function act(action){
  const r = await fetch("/api/action", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ action })
  });
  const j = await r.json();
  if(!j.ok){
    const box = document.getElementById("errBox");
    box.style.display = "block";
    box.textContent = j.error || "Action failed";
  }
  await refresh();
}

async function toggleFace(){
  const st = await getState();
  await fetch("/api/action", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ action:"set_face", enabled: !st.face_auto_enabled })
  });
  await refresh();
}

async function refresh(){
  const st = await getState();
  document.getElementById("pill").textContent =
    `Door: ${st.door_open ? "OPEN" : "CLOSED"} • LED: ${st.led_on ? "ON" : "OFF"} • Users: ${(st.users||[]).length}`;

  const doorV = document.getElementById("doorV");
  doorV.textContent = st.door_open ? "OPEN" : "CLOSED";
  doorV.className = "v " + (st.door_open ? "warn" : "good");

  const ledV = document.getElementById("ledV");
  ledV.textContent = st.led_on ? "ON" : "OFF";
  ledV.className = "v " + (st.led_on ? "good" : "bad");

  const whoV = document.getElementById("whoV");
  whoV.textContent = st.last_verified ? `${st.last_verified} (${(st.last_score||0).toFixed(3)})` : "—";
  whoV.className = "v " + (st.last_verified ? "good" : "bad");

  const bellV = document.getElementById("bellV");
  bellV.textContent = st.doorbell_pending ? "PENDING" : "NONE";
  bellV.className = "v " + (st.doorbell_pending ? "warn" : "good");

  const sw = document.getElementById("faceSw");
  sw.className = "switch" + (st.face_auto_enabled ? " on" : "");

  const bellBadge = document.getElementById("bellBadge");
  bellBadge.textContent = st.doorbell_pending ? "ACTION REQUIRED" : "IDLE";

  const bellNote = document.getElementById("bellNote");
  const snapWrap = document.getElementById("snapWrap");
  snapWrap.innerHTML = "";

  if(st.doorbell_pending){
    const t = st.doorbell_time ? new Date(st.doorbell_time*1000).toLocaleString() : "";
    bellNote.textContent = `Doorbell pressed. ${t}`;
    if(st.doorbell_image){
      const img = document.createElement("img");
      img.src = st.doorbell_image + "?ts=" + Date.now();
      snapWrap.appendChild(img);
    }
  } else {
    bellNote.textContent = "No event.";
  }

  const box = document.getElementById("errBox");
  if(st.error){
    box.style.display = "block";
    box.textContent = st.error;
  } else {
    box.style.display = "none";
  }
}
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""

app = Flask(__name__)

@app.get("/")
def home():
    return render_template_string(HTML)

@app.get("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            with frame_lock:
                jpg = latest_jpeg
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/state")
def api_state():
    with state_lock:
        return jsonify(dict(state))

@app.post("/api/action")
def api_action():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")

    try:
        if action == "open":
            door_open(reason="web")
        elif action == "close":
            door_close(reason="web")
        elif action == "entered":
            inside_entered()
        elif action == "led_on":
            led_set(True)
        elif action == "led_off":
            led_set(False)
        elif action == "ring":
            ring_bell()
        elif action == "reload":
            users = reload_known()
            with state_lock:
                state["users"] = users
            set_event("Users reloaded")
        elif action == "clear_bell":
            with state_lock:
                state["doorbell_pending"] = False
                state["doorbell_time"] = None
                state["doorbell_image"] = None
            set_event("Doorbell cleared")
        elif action == "snapshot":
            img_path = os.path.join(STATIC_DIR, "doorbell.jpg")
            ok = save_snapshot(img_path)
            with state_lock:
                state["doorbell_image"] = "/static/doorbell.jpg" if ok else None
                if ok and not state["doorbell_time"]:
                    state["doorbell_time"] = int(time.time())
            if not ok:
                set_error("Snapshot failed (no frame yet).")
        elif action == "set_face":
            enabled = bool(data.get("enabled"))
            with state_lock:
                state["face_auto_enabled"] = enabled
            set_event(f"Face auto-open set to {enabled}")
        else:
            return jsonify({"ok": False, "error": "Unknown action"}), 400

        return jsonify({"ok": True})
    except Exception as e:
        set_error(str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    with state_lock:
        state["users"] = reload_known()
        state["door_open"] = bool(getattr(door, "is_open", False))
        state["led_on"] = False

    print(f"project.py running on http://0.0.0.0:{APP_PORT}")
    print("If camera is busy, stop other scripts using the camera (like app.py).")
    try:
        app.run(host=APP_HOST, port=APP_PORT, threaded=True)
    finally:
        cleanup_all()
