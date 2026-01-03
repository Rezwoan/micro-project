import cv2 as cv
import numpy as np
import time

LIVE_MODEL = "/home/pi/project/models/liveness.onnx"

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

net = cv.dnn.readNetFromONNX(LIVE_MODEL)

cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

# Auto-detect which output index corresponds to "real" (assume first ~2s is a real face)
real_idx = None
calib_logits = []

t0 = time.time()
while time.time() - t0 < 2.0:
    ok, frame = cap.read()
    if not ok:
        continue
    # For this quick test, just center-crop as a placeholder.
    h, w = frame.shape[:2]
    s = min(h, w)
    crop = frame[h//2 - s//4:h//2 + s//4, w//2 - s//4:w//2 + s//4]

    blob = cv.dnn.blobFromImage(crop, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    logits = out.reshape(-1).astype(np.float32)
    if logits.size >= 2:
        calib_logits.append(logits[:2])

if calib_logits:
    mean_logits = np.mean(np.stack(calib_logits), axis=0)
    real_idx = int(np.argmax(mean_logits))

print("Calibrated real_idx =", real_idx)

# Now print live probability continuously
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    h, w = frame.shape[:2]
    s = min(h, w)
    crop = frame[h//2 - s//4:h//2 + s//4, w//2 - s//4:w//2 + s//4]

    blob = cv.dnn.blobFromImage(crop, scalefactor=1/255.0, size=(80, 80), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    logits = out.reshape(-1).astype(np.float32)[:2]
    p = softmax(logits)
    live_prob = float(p[real_idx]) if real_idx is not None else float(np.max(p))

    print(f"live_prob={live_prob:.3f}")
