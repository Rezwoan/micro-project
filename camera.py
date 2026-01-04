import cv2 as cv
import numpy as np
import os
import threading
import time
import datetime
import json
import door_module
import lighting_module

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
SNAP_DIR = os.path.join(STATIC_DIR, "snapshots")
PROFILES_FILE = os.path.join(DB_DIR, "profiles.json")

MODEL_YUNET = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
MODEL_SFACE = os.path.join(MODEL_DIR, "face_recognition_sface.onnx")

RECOG_THRESH = 0.55 

if not os.path.exists(SNAP_DIR): os.makedirs(SNAP_DIR)

class VideoCamera(object):
    def __init__(self):
        door_module.init()
        door_module.door_close()
        lighting_module.init()

        self.video = cv.VideoCapture(0)
        self.video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv.CAP_PROP_FPS, 30)

        self.detector = cv.FaceDetectorYN.create(MODEL_YUNET, "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv.FaceRecognizerSF.create(MODEL_SFACE, "")
        
        self.known_embeddings = {}
        self.profiles = {} 
        self.load_data()

        self.frame = None
        self.lock = threading.Lock()
        self.is_running = True
        
        self.last_faces_data = [] 
        self.current_subtitle = "SCANNING..."
        self.notifications = [] 
        
        self.doorbell_ringing = False
        self.last_ring_time = 0
        self.is_unlocked = False
        
        self.is_enrolling = False
        self.enroll_name = ""
        self.enroll_samples = []

        self.btn1_last = False
        self.btn2_last = False

        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def load_data(self):
        self.known_embeddings = {}
        if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
        for filename in os.listdir(DB_DIR):
            if filename.endswith(".npz"):
                name = os.path.splitext(filename)[0]
                try:
                    data = np.load(os.path.join(DB_DIR, filename))
                    if 'mean' in data: self.known_embeddings[name] = data['mean'].astype(np.float32)
                    elif 'embs' in data: self.known_embeddings[name] = data['embs'].mean(axis=0).astype(np.float32)
                except: pass
        
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE, 'r') as f:
                    self.profiles = json.load(f)
            except: self.profiles = {}

    def save_profile(self, name, l1, l2, unlock):
        self.profiles[name] = {"l1": l1, "l2": l2, "unlock": unlock}
        with open(PROFILES_FILE, 'w') as f:
            json.dump(self.profiles, f)

    def delete_face(self, name):
        path = os.path.join(DB_DIR, f"{name}.npz")
        if os.path.exists(path): os.remove(path)
        if name in self.known_embeddings: del self.known_embeddings[name]
        if name in self.profiles: del self.profiles[name]
        self.add_notification(f"Deleted User: {name}", "SYSTEM")

    def cosine_sim(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def add_notification(self, msg, type="INFO", img_path=None):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        # Unique ID for frontend deduplication
        uid = int(time.time() * 1000)
        self.notifications.insert(0, {
            'id': uid,
            'time': ts, 
            'msg': msg, 
            'type': type, 
            'img': img_path
        })
        self.notifications = self.notifications[:50] 

    def trigger_doorbell(self):
        self.doorbell_ringing = True
        self.last_ring_time = time.time()
        
        fname = f"ring_{int(time.time())}.jpg"
        save_path = os.path.join(SNAP_DIR, fname)
        with self.lock:
            if self.frame is not None:
                cv.imwrite(save_path, self.frame)
        
        self.add_notification("🔔 Doorbell Ringing!", "BELL", f"static/snapshots/{fname}")
        threading.Thread(target=door_module.play_buzzer_sequence).start()
        
    def unlock_door(self):
        if not self.is_unlocked:
            self.is_unlocked = True
            door_module.door_open() 
            self.add_notification("🔓 Door Unlocked", "UNLOCK")

    def lock_door(self):
        if self.is_unlocked:
            self.is_unlocked = False
            door_module.door_close()
            self.add_notification("🔒 Door Locked", "LOCK")

    def toggle_light(self, light_id):
        current = lighting_module.get_light_state(light_id)
        lighting_module.set_light(light_id, 0 if current else 1)

    def process_verified_user(self, name):
        p = self.profiles.get(name, {"l1": True, "l2": True, "unlock": True})
        lighting_module.set_light(1, 1 if p.get('l1') else 0)
        lighting_module.set_light(2, 1 if p.get('l2') else 0)
        
        if p.get('unlock', True):
            if not self.is_unlocked:
                self.unlock_door()
                self.add_notification(f"Verified: {name}", "VERIFIED")
        else:
            if not any(n['msg'] == f"{name} Arrived (Locked)" and time.time() - self.last_ring_time < 10 for n in self.notifications[:3]):
                self.add_notification(f"{name} Arrived (Locked)", "ALERT")
    
    def start_enrollment(self, name):
        self.is_enrolling = True
        self.enroll_name = name
        self.enroll_samples = []
        self.add_notification(f"Learning: {name}", "SYSTEM")

    def _update(self):
        frame_count = 0
        while self.is_running:
            success, frame = self.video.read()
            if not success: time.sleep(0.01); continue
            
            frame = cv.flip(frame, 1)
            frame_count += 1
            
            # Poll Buttons
            if door_module.is_lock_button_pressed():
                if "WELCOME" not in self.current_subtitle: self.lock_door()
            if door_module.is_doorbell_button_pressed():
                if not self.doorbell_ringing: self.trigger_doorbell()
            
            b1 = lighting_module.is_btn_pressed(1)
            if b1 and not self.btn1_last: self.toggle_light(1)
            self.btn1_last = b1
            b2 = lighting_module.is_btn_pressed(2)
            if b2 and not self.btn2_last: self.toggle_light(2)
            self.btn2_last = b2
            
            # AI Logic
            if frame_count % 4 == 0:
                h, w = frame.shape[:2]
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(frame)
                
                temp_faces_data = []
                detected_names = []

                if faces is not None:
                    # --- STRICT ENROLLMENT LOGIC ---
                    if self.is_enrolling:
                        if len(faces) > 1:
                            self.current_subtitle = "ERROR: TOO MANY FACES!"
                            self.is_enrolling = False
                            self.enroll_samples = []
                            self.add_notification("Enroll Failed: Too many faces", "ALERT")
                        else:
                            # Valid single face
                            face = faces[0]
                            box = face[0:4].astype(np.int32)
                            aligned_face = self.recognizer.alignCrop(frame, face)
                            feat = self.recognizer.feature(aligned_face)
                            curr_emb = feat.flatten().astype(np.float32)
                            
                            self.enroll_samples.append(curr_emb)
                            self.current_subtitle = f"HOLD STILL: {len(self.enroll_samples)}/15"
                            temp_faces_data.append({'box': box, 'color': (0, 255, 255)}) # Yellow

                            if len(self.enroll_samples) >= 15:
                                mean_emb = np.mean(self.enroll_samples, axis=0)
                                np.savez_compressed(os.path.join(DB_DIR, f"{self.enroll_name}.npz"), mean=mean_emb)
                                self.known_embeddings[self.enroll_name] = mean_emb
                                self.is_enrolling = False
                                self.add_notification(f"Registered: {self.enroll_name}", "SYSTEM")
                                self.load_data()

                    else:
                        # Normal Recognition
                        for face in faces:
                            box = face[0:4].astype(np.int32)
                            aligned_face = self.recognizer.alignCrop(frame, face)
                            feat = self.recognizer.feature(aligned_face)
                            curr_emb = feat.flatten().astype(np.float32)

                            best_name = "Unknown"
                            max_score = -1.0
                            for name, db_emb in self.known_embeddings.items():
                                score = self.cosine_sim(curr_emb, db_emb)
                                if score > max_score:
                                    max_score = score
                                    best_name = name
                            
                            if max_score > RECOG_THRESH:
                                detected_names.append(best_name)
                                color = (0, 255, 0)
                                self.process_verified_user(best_name)
                            else:
                                color = (0, 0, 255)
                            
                            temp_faces_data.append({'box': box, 'color': color})

                        if len(detected_names) > 0:
                            self.current_subtitle = f"WELCOME {', '.join(detected_names)}"
                        else:
                            self.current_subtitle = "UNKNOWN VISITOR"
                
                else:
                    # No Faces Found
                    if self.is_enrolling:
                        self.current_subtitle = "NO FACE DETECTED"
                        # Don't cancel, just wait
                    else:
                        self.current_subtitle = "SCANNING..."
                
                self.last_faces_data = temp_faces_data

            # Draw
            for item in self.last_faces_data:
                x, y, bw, bh = item['box']
                cv.rectangle(frame, (x, y), (x+bw, y+bh), item['color'], 2)
            
            h, w = frame.shape[:2]
            cv.rectangle(frame, (0, h-50), (w, h), (0,0,0), -1)
            
            display_text = self.current_subtitle
            text_color = (255, 255, 255)
            
            if self.is_unlocked:
                if "WELCOME" in self.current_subtitle:
                     display_text += " - OPEN"
                     text_color = (0, 255, 0)
                else:
                    display_text = "DOOR UNLOCKED"
                    text_color = (0, 255, 0)

            text_size = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv.putText(frame, display_text, (text_x, h - 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            with self.lock:
                self.frame = frame
                if self.doorbell_ringing and (time.time() - self.last_ring_time > 5):
                    self.doorbell_ringing = False
            
            time.sleep(1/24)

    def get_frame(self):
        with self.lock:
            if self.frame is None: return None
            ret, jpeg = cv.imencode('.jpg', self.frame)
            return jpeg.tobytes()
            
    def get_status(self):
        with self.lock:
            return {
                "notifications": self.notifications,
                "ringing": self.doorbell_ringing,
                "is_unlocked": self.is_unlocked,
                "l1": lighting_module.get_light_state(1),
                "l2": lighting_module.get_light_state(2),
                "users": list(self.known_embeddings.keys()),
                "profiles": self.profiles
            }