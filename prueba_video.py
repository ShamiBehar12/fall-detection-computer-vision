# annotate_fallnet_fixed.py
import cv2, torch, numpy as np, mediapipe as mp
from fallnet_arch import FallNet

# ───── CONFIG ─────
MODEL_PATH   = r"C:\Users\lenovo\Desktop\MVP_FLO\fallnet_best.pth"
INPUT_VIDEO  = r"C:\Users\lenovo\Desktop\MVP_FLO\video_caida_jaime.mp4"
OUTPUT_VIDEO = r"C:\Users\lenovo\Desktop\MVP_FLO\video_caida_jaime_output.mp4"
ROTATE_MODE  = 1          # 0-sin giro | 1-CW | 2-CCW | 3-180°
SPEED_FACTOR = 0.8
WINDOW, THRESHOLD = 30, 0.5
POINT_OK, POINT_ALERT = (0,255,0), (0,0,255)
LINE_COLOR = (200,200,200)
# ──────────────────

def apply_rotation(img, mode):
    if mode == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if mode == 2:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    return img

print("→ Cargando pesos")
net = FallNet(num_layers=3, hidden_size=128, dropout=0.3)
net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
net.eval()

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=False)
CONNS   = mp_pose.POSE_CONNECTIONS

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("No se puede abrir", INPUT_VIDEO)

fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ajusta dimensiones tras el giro
if ROTATE_MODE in (1,2):
    out_w, out_h = h0, w0
elif ROTATE_MODE == 3:
    out_w, out_h = w0, h0
else:
    out_w, out_h = w0, h0

fps_out = max(1, int(fps_in * SPEED_FACTOR))
writer  = cv2.VideoWriter(
            OUTPUT_VIDEO,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_out, (out_w, out_h))

buffer, label = [], "…"
print(f"Procesando {INPUT_VIDEO} → {OUTPUT_VIDEO}  "
      f"({out_w}×{out_h}, {fps_in:.1f} fps → {fps_out} fps, rot={ROTATE_MODE})")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = apply_rotation(frame, ROTATE_MODE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    # keypoints
    kp = np.zeros(66, np.float32)
    if res.pose_landmarks:
        kp = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark], np.float32).flatten()
    buffer.append(kp); buffer[:] = buffer[-WINDOW:]

    # inferencia
    if len(buffer) == WINDOW:
        window = torch.from_numpy(np.stack(buffer)).unsqueeze(0)
        prob   = torch.sigmoid(net(window))[0].item()
        label  = "CAIDA" if prob > THRESHOLD else "NO CAIDA"

    # overlay
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        for a,b in CONNS:
            xa,ya = int(lm[a].x*out_w), int(lm[a].y*out_h)
            xb,yb = int(lm[b].x*out_w), int(lm[b].y*out_h)
            cv2.line(frame,(xa,ya),(xb,yb), LINE_COLOR,2)
        color_pt = POINT_ALERT if label=="CAIDA" else POINT_OK
        for p in lm:
            x,y = int(p.x*out_w), int(p.y*out_h)
            cv2.circle(frame,(x,y),4,color_pt,-1)

    cv2.putText(frame, label, (70,200), cv2.FONT_HERSHEY_SIMPLEX, 2,
                POINT_ALERT if label=="CAIDA" else POINT_OK, 3)

    writer.write(frame)

cap.release(); writer.release()
print("✔ Guardado en:", OUTPUT_VIDEO)
