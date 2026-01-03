from flask import Flask, Response, redirect, url_for, render_template_string, request
from gpiozero import LED
import threading
import time
import os
import subprocess
import cv2
import atexit

# =========================
# Config
# =========================
LED_GPIO = 21                # Pin 40
CAM_INDEX = 0                # Use 0 for /dev/video0, 1 for /dev/video1
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_JPEG_QUALITY = 80

SNAP_WIDTH = 1280
SNAP_HEIGHT = 720
SNAP_JPEG_QUALITY = 95

PHOTO_PATH = "static/latest.jpg"

# =========================
# App + Hardware
# =========================
app = Flask(__name__)
led = LED(LED_GPIO)

# Blink control
blink_thread = None
blink_stop = threading.Event()
blink_lock = threading.Lock()
blink_interval = 1.0

# Camera shared objects
cam_lock = threading.Lock()
cap = None

# UI state
photo_ts = int(time.time())
last_error = None


def ensure_static_dir():
    os.makedirs(os.path.dirname(PHOTO_PATH), exist_ok=True)


def set_error(msg: str | None):
    global last_error
    last_error = msg


# -------------------------
# Camera: open/reopen safely
# -------------------------
def open_camera():
    """Open camera if not opened; return True if ready."""
    global cap
    if cap is not None and cap.isOpened():
        return True

    # Try open fresh
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        set_error("Camera open failed. Try changing CAM_INDEX to 1 or check /dev/video0.")
        return False

    # Set stream size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    return True


def close_camera():
    global cap
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    cap = None


atexit.register(close_camera)


# -------------------------
# MJPEG generator
# -------------------------
def gen_frames():
    global cap
    while True:
        with cam_lock:
            ok = open_camera()
            if not ok:
                # If camera not available, wait a bit and retry
                frame = None
            else:
                ret, frame = cap.read()

        if frame is None or not ret:
            time.sleep(0.1)
            continue

        # Encode JPEG
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
        if not ok:
            continue

        jpg = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


# -------------------------
# Snapshot (best reliability: fswebcam)
# -------------------------
def take_snapshot():
    """
    Take a snapshot and overwrite static/latest.jpg
    Uses fswebcam for simple, reliable capture.
    """
    ensure_static_dir()

    # Release OpenCV camera while fswebcam grabs it (prevents device busy issues)
    with cam_lock:
        close_camera()

    cmd = [
        "fswebcam",
        "-d", f"/dev/video{CAM_INDEX}",
        "--no-banner",
        "-r", f"{SNAP_WIDTH}x{SNAP_HEIGHT}",
        "--jpeg", str(SNAP_JPEG_QUALITY),
        PHOTO_PATH
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, None
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout).decode(errors="ignore")
        return False, (err[:800] if err else "Snapshot failed (unknown error).")


# -------------------------
# Blinking
# -------------------------
def blink_worker():
    global blink_interval
    while not blink_stop.is_set():
        with blink_lock:
            interval = float(blink_interval)

        led.on()
        if blink_stop.wait(interval):
            break
        led.off()
        if blink_stop.wait(interval):
            break


def blinking_is_running():
    global blink_thread
    return blink_thread is not None and blink_thread.is_alive()


def start_blinking():
    global blink_thread
    if blinking_is_running():
        return
    blink_stop.clear()
    blink_thread = threading.Thread(target=blink_worker, daemon=True)
    blink_thread.start()


def stop_blinking():
    blink_stop.set()


# =========================
# UI (Cyberpunk, mobile-first)
# =========================
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pi Cyber Control</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

    :root{
      --bg:#07080c;
      --panel:#101423;
      --panel2:#0c0f1a;
      --text:#d9e2ff;
      --muted:#8ea1ff;

      --neon:#66fcf1;
      --neon2:#ff79c6;
      --warn:#ffb86c;
      --danger:#ff5555;

      --stroke:rgba(102,252,241,.25);
      --stroke2:rgba(255,121,198,.25);

      --radius:18px;
      --shadow: 0 0 22px rgba(102,252,241,.18);
      --shadow2: 0 0 22px rgba(255,121,198,.14);
    }

    *{ box-sizing:border-box; }
    body{
      margin:0;
      background:
        radial-gradient(900px 600px at 10% -10%, rgba(102,252,241,.18), transparent 60%),
        radial-gradient(900px 600px at 110% 10%, rgba(255,121,198,.14), transparent 55%),
        linear-gradient(180deg, #05060a, #07080c);
      color:var(--text);
      font-family: "Orbitron", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      display:flex;
      justify-content:center;
      padding:16px;
    }

    .wrap{
      width:100%;
      max-width:540px;
    }

    .top{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      padding:14px 14px 18px;
    }

    .title{
      font-size:20px;
      letter-spacing:.06em;
      font-weight:800;
      color:var(--neon);
      text-shadow: 0 0 10px rgba(102,252,241,.35);
    }

    .sub{
      font-size:12px;
      color:var(--muted);
      margin-top:6px;
      letter-spacing:.08em;
    }

    .pill{
      padding:10px 12px;
      border:1px solid var(--stroke);
      border-radius:999px;
      background: rgba(16,20,35,.55);
      box-shadow: var(--shadow);
      font-size:12px;
      color:var(--neon);
      white-space:nowrap;
    }

    .grid{
      display:grid;
      gap:14px;
    }

    .card{
      border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(16,20,35,.9), rgba(12,15,26,.92));
      border: 1px solid rgba(102,252,241,.18);
      box-shadow: var(--shadow);
      overflow:hidden;
    }

    .card .hd{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      padding:14px 14px 10px;
      border-bottom: 1px solid rgba(102,252,241,.10);
      background: rgba(7,8,12,.25);
    }

    .card .hd .h{
      font-weight:800;
      font-size:13px;
      letter-spacing:.12em;
      color:var(--neon);
    }

    .badge{
      font-size:12px;
      padding:8px 10px;
      border-radius:999px;
      border:1px solid rgba(255,121,198,.20);
      background: rgba(255,121,198,.08);
      color: var(--neon2);
      box-shadow: var(--shadow2);
      white-space:nowrap;
    }

    .body{
      padding:14px;
    }

    .btnrow{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap:10px;
    }

    .btn{
      border:0;
      padding:14px 12px;
      border-radius:14px;
      font-family: inherit;
      font-weight:800;
      letter-spacing:.08em;
      cursor:pointer;
      transition: transform .08s ease, filter .12s ease, box-shadow .12s ease;
      color:#061018;
      background: var(--neon2);
      box-shadow: 0 0 0 1px rgba(255,121,198,.18), 0 10px 26px rgba(255,121,198,.18);
    }
    .btn:active{ transform: translateY(1px) scale(.99); }
    .btn:hover{ filter: brightness(1.06); }

    .btn.on{ background: var(--neon); box-shadow: 0 0 0 1px rgba(102,252,241,.22), 0 10px 26px rgba(102,252,241,.16); }
    .btn.off{ background: var(--danger); box-shadow: 0 0 0 1px rgba(255,85,85,.22), 0 10px 26px rgba(255,85,85,.14); }
    .btn.warn{ background: var(--warn); box-shadow: 0 0 0 1px rgba(255,184,108,.22), 0 10px 26px rgba(255,184,108,.14); }

    .full{ grid-column: 1 / -1; }

    .statline{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:10px;
      padding:10px 12px;
      border-radius:14px;
      border:1px solid rgba(102,252,241,.12);
      background: rgba(6,8,14,.35);
      margin-bottom:12px;
    }
    .statline .k{ color: var(--muted); font-size:12px; letter-spacing:.10em; }
    .statline .v{ font-weight:800; font-size:12px; letter-spacing:.10em; }

    .good{ color: var(--neon); }
    .bad{ color: var(--danger); }
    .mid{ color: var(--warn); }

    .input{
      display:flex;
      gap:10px;
      align-items:center;
      margin: 10px 0 12px;
    }
    input[type="number"]{
      flex:1;
      padding:14px 12px;
      border-radius:14px;
      border:1px solid rgba(102,252,241,.22);
      background: rgba(7,8,12,.35);
      color: var(--text);
      font-family: inherit;
      font-weight:700;
      letter-spacing:.08em;
      outline:none;
    }

    .video{
      width:100%;
      aspect-ratio: 16/9;
      background: #000;
      border-radius: 16px;
      border: 1px solid rgba(255,121,198,.22);
      box-shadow: var(--shadow2);
      overflow:hidden;
    }
    .video img{
      width:100%;
      height:100%;
      object-fit: cover;
      display:block;
    }

    .snap img{
      width:100%;
      border-radius: 16px;
      border: 1px solid rgba(102,252,241,.20);
      box-shadow: var(--shadow);
      display:block;
      margin-top:10px;
    }

    .err{
      margin-top:12px;
      padding:12px;
      border-radius:14px;
      border:1px solid rgba(255,85,85,.25);
      background: rgba(255,85,85,.08);
      color: #ffd6d6;
      font-size:12px;
      letter-spacing:.06em;
      white-space: pre-wrap;
    }

    @media (max-width: 420px){
      .title{ font-size:18px; }
      .btn{ padding:13px 10px; }
    }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="title">CYBER CONTROL</div>
        <div class="sub">GPIO + Live Video • Pi 5</div>
      </div>
      <div class="pill">GPIO{{ gpio }} • PIN40</div>
    </div>

    <div class="grid">

      <div class="card">
        <div class="hd">
          <div class="h">LIVE VIDEO</div>
          <div class="badge">MJPEG STREAM</div>
        </div>
        <div class="body">
          <div class="video">
            <img src="{{ url_for('video_feed') }}" alt="Live stream">
          </div>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <div class="h">LED CONTROL</div>
          <div class="badge">{{ "ON" if led_on else "OFF" }}</div>
        </div>
        <div class="body">
          <div class="statline">
            <div class="k">STATUS</div>
            <div class="v {{ 'good' if led_on else 'bad' }}">{{ "⚡ ON" if led_on else "⭘ OFF" }}</div>
          </div>

          <div class="btnrow">
            <form action="/on" method="post"><button class="btn on" type="submit">POWER ON</button></form>
            <form action="/off" method="post"><button class="btn off" type="submit">POWER OFF</button></form>
            <form class="full" action="/toggle" method="post"><button class="btn" type="submit">TOGGLE</button></form>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <div class="h">BLINK ENGINE</div>
          <div class="badge">{{ "RUNNING" if blinking else "STOPPED" }}</div>
        </div>
        <div class="body">
          <div class="statline">
            <div class="k">MODE</div>
            <div class="v {{ 'mid' if blinking else 'bad' }}">{{ "BLINK" if blinking else "IDLE" }}</div>
          </div>

          <form action="/blink/start" method="post">
            <div class="input">
              <input type="number" name="interval" min="0.05" step="0.05" value="{{ interval }}" />
            </div>
            <button class="btn warn full" type="submit">START BLINK</button>
          </form>

          <form action="/blink/stop" method="post">
            <button class="btn off full" type="submit">STOP BLINK</button>
          </form>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <div class="h">SNAPSHOT</div>
          <div class="badge">REPLACES OLD</div>
        </div>
        <div class="body snap">
          <form action="/photo" method="post">
            <button class="btn full" type="submit">📸 TAKE PHOTO</button>
          </form>

          {% if photo_exists %}
            <img src="/static/latest.jpg?ts={{ photo_ts }}" alt="Latest photo">
          {% else %}
            <div class="statline" style="margin-top:12px;">
              <div class="k">INFO</div>
              <div class="v mid">NO PHOTO YET</div>
            </div>
          {% endif %}

          {% if error %}
            <div class="err">{{ error }}</div>
          {% endif %}
        </div>
      </div>

    </div>
  </div>
</body>
</html>
"""


# =========================
# Routes
# =========================
@app.get("/")
def index():
    return render_template_string(
        HTML,
        gpio=LED_GPIO,
        led_on=led.is_lit,
        blinking=blinking_is_running(),
        interval=blink_interval,
        photo_ts=photo_ts,
        photo_exists=os.path.exists(PHOTO_PATH),
        error=last_error
    )


@app.get("/video_feed")
def video_feed():
    # Important: this must stream continuously
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/on")
def route_on():
    try:
        set_error(None)
        stop_blinking()
        led.on()
    except Exception as e:
        set_error(f"LED ON failed: {e}")
    return redirect(url_for("index"))


@app.post("/off")
def route_off():
    try:
        set_error(None)
        stop_blinking()
        led.off()
    except Exception as e:
        set_error(f"LED OFF failed: {e}")
    return redirect(url_for("index"))


@app.post("/toggle")
def route_toggle():
    try:
        set_error(None)
        stop_blinking()
        led.toggle()
    except Exception as e:
        set_error(f"LED TOGGLE failed: {e}")
    return redirect(url_for("index"))


@app.post("/blink/start")
def route_blink_start():
    global blink_interval
    try:
        set_error(None)
        val = float(request.form.get("interval", "1.0"))
        # clamp
        val = max(0.05, min(val, 10.0))
        with blink_lock:
            blink_interval = val
        start_blinking()
    except Exception as e:
        set_error(f"BLINK START failed: {e}")
    return redirect(url_for("index"))


@app.post("/blink/stop")
def route_blink_stop():
    try:
        set_error(None)
        stop_blinking()
        led.off()
    except Exception as e:
        set_error(f"BLINK STOP failed: {e}")
    return redirect(url_for("index"))


@app.post("/photo")
def route_photo():
    global photo_ts
    try:
        set_error(None)
        ok, err = take_snapshot()
        if not ok:
            set_error(err or "Snapshot failed.")
        else:
            photo_ts = int(time.time())  # cache-bust
    except Exception as e:
        set_error(f"SNAPSHOT failed: {e}")
    return redirect(url_for("index"))


if __name__ == "__main__":
    ensure_static_dir()
    # threaded=True lets the stream + button requests work together
    app.run(host="0.0.0.0", port=5000, threaded=True)
