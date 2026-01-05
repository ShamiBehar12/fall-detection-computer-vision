# mvp_fallnet.py
# deps: fastapi uvicorn[standard] opencv-python mediapipe torch torchvision

import cv2, threading, time, numpy as np, torch, mediapipe as mp
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fallnet_arch import FallNet

# ───────── CONFIG ─────────
MODEL_PATH = r"C:\Users\lenovo\Desktop\MVP_FLO\fallnet_best.pth"
CAM_URL    = "http://192.168.43.243:8080/video"
FRAME_W, FRAME_H = 640, 360
WINDOW, THRESHOLD = 30, 0.5
POINT_OK, POINT_ALERT = (0, 255, 0), (0, 0, 255)
LINE_COLOR = (200, 200, 200)
# ──────────────────────────

print(">>> Cargando pesos...", flush=True)
model = FallNet(num_layers=3, hidden_size=128, dropout=0.3)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
fallnet = model.eval()
print("Paráms totales:", round(sum(p.numel() for p in fallnet.parameters())/1e6, 2), "M")

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=False)
CONNS   = mp_pose.POSE_CONNECTIONS

buffer: list[np.ndarray] = []
latest_jpg: Optional[bytes] = None
label = "…"

def grab_and_process():
    global latest_jpg, label
    print(">>> Hilo de captura arrancando", flush=True)
    cap = cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():
        raise RuntimeError(f"No abre la URL {CAM_URL}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05);  continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = pose.process(rgb)

        # ---------- keypoints 2D ----------
        kp = np.zeros(66, np.float32)
        if res.pose_landmarks:
            kp = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark],
                          dtype=np.float32).flatten()
        buffer.append(kp);  buffer[:] = buffer[-WINDOW:]

        # ---------- inferencia ------------
        if len(buffer) == WINDOW:
            window = torch.from_numpy(np.stack(buffer)).unsqueeze(0)
            prob   = torch.sigmoid(fallnet(window))[0].item()
            label  = "CAIDA" if prob > THRESHOLD else "NO CAIDA"

        # ---------- overlay ---------------
        if res.pose_landmarks:
            lm_list = res.pose_landmarks.landmark
            # líneas
            for a, b in CONNS:
                xa, ya = int(lm_list[a].x*FRAME_W), int(lm_list[a].y*FRAME_H)
                xb, yb = int(lm_list[b].x*FRAME_W), int(lm_list[b].y*FRAME_H)
                cv2.line(frame, (xa, ya), (xb, yb), LINE_COLOR, 2)
            # puntos
            for lm in lm_list:
                x, y = int(lm.x*FRAME_W), int(lm.y*FRAME_H)
                cv2.circle(frame, (x, y), 4,
                           POINT_ALERT if label=="CAIDA" else POINT_OK, -1)

        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    POINT_ALERT if label=="CAIDA" else POINT_OK, 3)

        latest_jpg = cv2.imencode('.jpg', frame)[1].tobytes()

# ───────── FastAPI ─────────
app = FastAPI()

def stream():
    while True:
        if latest_jpg is None:
            time.sleep(0.05); continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               latest_jpg + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(stream(),
        media_type="multipart/x-mixed-replace; boundary=frame")

# ───────── MAIN ─────────
if __name__ == "__main__":
    threading.Thread(target=grab_and_process, daemon=True).start()
    print(">>> Servidor en  http://localhost:8000/video_feed", flush=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
