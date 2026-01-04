from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from camera import VideoCamera
import time

app = Flask(__name__)
app.secret_key = 'super_secret'
video_stream = VideoCamera()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.02)

# ... (Routes for /, login, doorbell, video_feed stay the same) ...
@app.route('/')
def index(): return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('email') == "pi@pi.pi" and request.form.get('password') == "pipipipi":
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/doorbell')
def doorbell(): return render_template('doorbell.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API ---
@app.route('/api/status')
def get_status(): return jsonify(video_stream.get_status())

@app.route('/api/enroll', methods=['POST'])
def enroll_face():
    name = request.json.get('name')
    if name:
        video_stream.start_enrollment(name)
        return jsonify({"status": "started"})
    return jsonify({"status": "error"}), 400

@app.route('/api/ring', methods=['POST'])
def ring_bell():
    video_stream.trigger_doorbell()
    return jsonify({"status": "ringing"})

@app.route('/api/unlock', methods=['POST'])
def unlock():
    video_stream.unlock_door()
    return jsonify({"status": "unlocked"})

@app.route('/api/lock', methods=['POST'])
def lock():
    video_stream.lock_door()
    return jsonify({"status": "locked"})

@app.route('/api/light/toggle', methods=['POST'])
def toggle_light():
    lid = request.json.get('id')
    video_stream.toggle_light(int(lid))
    return jsonify({"status": "ok"})

@app.route('/api/profile/save', methods=['POST'])
def save_profile():
    d = request.json
    video_stream.save_profile(d['name'], d['l1'], d['l2'], d['unlock'])
    return jsonify({"status": "saved"})

@app.route('/api/profile/delete', methods=['POST'])
def delete_profile():
    d = request.json
    video_stream.delete_face(d['name'])
    return jsonify({"status": "deleted"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)