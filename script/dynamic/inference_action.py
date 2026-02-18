import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# ==========================
# PATHS
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'model_action')
MODEL_PATH = os.path.join(MODEL_DIR, 'model_action.p')

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found")
    exit()

print("Using model:", MODEL_PATH)

# ==========================
# LOAD MODEL CHECKPOINT
# ==========================
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
unique_labels = checkpoint['unique_labels']
idx_to_label = {i: lbl for i, lbl in enumerate(unique_labels)}
input_size = checkpoint['input_size']

# ==========================
# MODEL DEFINITION
# ==========================
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = SignLSTM(input_size=input_size,
                 hidden_size=256,
                 num_layers=3,
                 num_classes=len(unique_labels))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ==========================
# CAMERA SETUP (half body)
# ==========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.5,
                       max_num_hands=2)

SEQUENCE_LENGTH = 12
MOTION_THRESHOLD = 0.03
CONF_THRESHOLD = 0.4
DISPLAY_HOLD_FRAMES = 30

frame_buffer = []
display_label = ""
display_timer = 0

print("\nStarted – press 'q' to quit\n")

# ==========================
# HELPER: get bounding box
# ==========================
def hand_bbox(hand_landmarks, frame_shape):
    H, W = frame_shape[:2]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    x1, y1 = int(min(xs) * W), int(min(ys) * H)
    x2, y2 = int(max(xs) * W), int(max(ys) * H)
    return x1, y1, x2, y2

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_frame_landmarks = []

    # ==========================
    # HAND DETECTION & BBOX
    # ==========================
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Bounding box
            x1, y1, x2, y2 = hand_bbox(hand_landmarks, frame.shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Normalize landmarks for model
            wrist_x, wrist_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
            ref_x, ref_y = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
            hand_size = np.hypot(ref_x - wrist_x, ref_y - wrist_y) + 1e-6

            data_aux = []
            for lm in hand_landmarks.landmark:
                dx = (lm.x - wrist_x) / hand_size
                dy = (lm.y - wrist_y) / hand_size
                data_aux.append(dx)
                data_aux.append(dy)
            current_frame_landmarks.append(np.array(data_aux))

    # ==========================
    # HANDLE 2 HANDS
    # ==========================
    if len(current_frame_landmarks) == 2:
        frame_features = np.concatenate(current_frame_landmarks)
    elif len(current_frame_landmarks) == 1:
        frame_features = np.concatenate([current_frame_landmarks[0], np.zeros(input_size//2)])
    else:
        frame_features = np.zeros(input_size)

    frame_buffer.append(frame_features)
    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer = frame_buffer[-SEQUENCE_LENGTH:]

    # ==========================
    # MOTION CHECK
    # ==========================
    has_motion = False
    avg_motion = 0.0
    if len(frame_buffer) >= 2:
        diffs = [np.abs(frame_buffer[i] - frame_buffer[i-1]).mean() for i in range(1, len(frame_buffer))]
        avg_motion = np.mean(diffs)
        has_motion = avg_motion > MOTION_THRESHOLD

    # ==========================
    # PREDICTION
    # ==========================
    if len(frame_buffer) == SEQUENCE_LENGTH and has_motion:
        seq = np.array([frame_buffer], dtype=np.float32)
        seq_tensor = torch.tensor(seq)
        with torch.no_grad():
            out = model(seq_tensor)
            probs = torch.softmax(out, dim=1)[0].numpy()
            conf = np.max(probs)
            pred_idx = np.argmax(probs)
            pred_label = idx_to_label.get(pred_idx, "?")

        if conf >= CONF_THRESHOLD:
            display_label = pred_label
            display_timer = DISPLAY_HOLD_FRAMES

        prob_dict = {lbl: round(p,3) for lbl,p in zip(unique_labels, probs)}
        print(f"Probs: {prob_dict} | Conf: {conf:.3f} → {pred_label} | Motion avg: {avg_motion:.4f}")

    # ==========================
    # DISPLAY
    # ==========================
    text_size = 1.0  # smaller
    text_thick = 2
    if display_timer > 0:
        cv2.putText(frame, f"{display_label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (80,255,120), text_thick)
        display_timer -= 1
    else:
        cv2.putText(frame, "No sign", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,165,255), text_thick)

    cv2.imshow("Dynamic Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
