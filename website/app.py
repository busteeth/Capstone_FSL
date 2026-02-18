import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
import time  # for timing/sleep

# ────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.p')
DATA_PATH = os.path.join(BASE_DIR, 'model', 'data.v1.2.pickle') #palitaan pag nag update

# Load model & labels
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

with open(DATA_PATH, 'rb') as f:
    data_dict = pickle.load(f)
    unique_labels = sorted(list(set(data_dict['labels'])))
    labels_dict = {i: label for i, label in enumerate(unique_labels)}

print("Loaded labels:", unique_labels)  # debug

# ────────────────────────────────────────────────
# CAMERA – LOW LATENCY SETTINGS
# ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# IMPORTANT: lower resolution + minimal buffer = much less lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)   # try 480 or 320 if still laggy
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # ← critical for low latency

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# FASTEST MediaPipe config
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,                    # lite = fastest
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    max_num_hands=2                       # single hand = faster
)

# ────────────────────────────────────────────────
# TUNING
# ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.84
ALPHA_SMOOTH    = 0.68
BOX_ALPHA       = 0.55
MIN_STABLE_FRAMES = 4

hand_trackers = {}
last_stable = {"sign": "—", "confidence": 0.0}

app = Flask(__name__)

def gen_frames():
    frame_skip_counter = 0
    target_interval = 1.0 / 25.0  # aim ~25 fps
    last_time = time.time()

    while True:
        # Flush old frames to reduce buffer lag
        for _ in range(2):
            cap.grab()

        ret, frame = cap.retrieve()
        if not ret:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_time < target_interval and frame_skip_counter < 2:
            frame_skip_counter += 1
            continue  # skip to keep pace
        frame_skip_counter = 0
        last_time = now

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_hands = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                current_hands.append(idx)

                handedness = "Unknown"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label

                # Bounding box...
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x1 = max(0, int(min(x_coords) * W) - 30)
                y1 = max(0, int(min(y_coords) * H) - 30)
                x2 = min(W, int(max(x_coords) * W) + 30)
                y2 = min(H, int(max(y_coords) * H) + 30)
                curr_box = (x1, y1, x2, y2)

                if idx not in hand_trackers:
                    hand_trackers[idx] = {
                        'smoothed_box': curr_box,
                        'smoothed_probs': None,
                        'stable_label': "",
                        'stable_count': 0,
                        'last_raw_label': "",
                    }

                tracker = hand_trackers[idx]

                # Smooth box
                sb = tracker['smoothed_box']
                tracker['smoothed_box'] = tuple(
                    int(BOX_ALPHA * c + (1 - BOX_ALPHA) * p) for c, p in zip(curr_box, sb)
                )

                # Normalize landmarks
                landmarks = hand_landmarks.landmark
                wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
                ref_x, ref_y = landmarks[12].x, landmarks[12].y
                hand_size = np.hypot(ref_x - wrist_x, ref_y - wrist_y) + 1e-8

                data_aux = []
                for lm in landmarks:
                    dx = lm.x - wrist_x
                    dy = lm.y - wrist_y
                    if handedness == "Right":
                        dx = -dx
                    data_aux.extend([dx / hand_size, dy / hand_size])

                current_probs = model.predict_proba([np.array(data_aux)])[0]

                sp = tracker['smoothed_probs']
                if sp is None:
                    sp = current_probs.copy()
                else:
                    sp = ALPHA_SMOOTH * sp + (1 - ALPHA_SMOOTH) * current_probs
                tracker['smoothed_probs'] = sp

                pred_idx = np.argmax(sp)
                conf = sp[pred_idx]
                raw_label = labels_dict.get(pred_idx, "?")

                if conf >= CONFIDENCE_THRESHOLD:
                    if raw_label == tracker['last_raw_label']:
                        tracker['stable_count'] += 1
                    else:
                        tracker['stable_count'] = 1
                        tracker['last_raw_label'] = raw_label

                    if tracker['stable_count'] >= MIN_STABLE_FRAMES:
                        tracker['stable_label'] = raw_label
                        global last_stable
                        last_stable = {"sign": raw_label, "confidence": round(conf * 100, 1)}
                else:
                    tracker['stable_count'] = 0
                    tracker['last_raw_label'] = ""
                    tracker['stable_label'] = ""

                # Draw
                x1, y1, x2, y2 = tracker['smoothed_box']
                color = (0, 220, 100) if tracker['stable_label'] else (100, 100, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, handedness[:5], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if tracker['stable_label']:
                    cv2.putText(frame, tracker['stable_label'],
                                (x2 - 60, y1 + 35),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (80, 255, 120), 2)

        # Clean disappeared hands
        to_remove = [k for k in hand_trackers if k not in current_hands]
        for k in to_remove:
            del hand_trackers[k]

        if not any(t['stable_label'] for t in hand_trackers.values()):
            last_stable = {"sign": "—", "confidence": 0.0}

        # Encode with LOWER quality → faster send + less mobile buffering
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # 70-75 is sweet spot
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_prediction')
def get_prediction():
    global last_stable
    return jsonify(last_stable)


if __name__ == "__main__":
    # debug=False in production, threaded=True helps a bit
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)