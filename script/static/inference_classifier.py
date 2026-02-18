import os
import pickle
import cv2
import numpy as np
import mediapipe as mp


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(BASE_DIR, 'model')

MODEL_PATH = os.path.join(MODEL_DIR, 'model.p')
DATA_PATH  = os.path.join(MODEL_DIR, 'data.v1.2.pickle') #palitaan pag nag update


try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
    print("Model loaded successfully from:", MODEL_PATH)
except FileNotFoundError:
    print("Error: model.p not found at:", MODEL_PATH)
    exit()

try:
    with open(DATA_PATH, 'rb') as f:
        data_dict = pickle.load(f)
        unique_labels = sorted(list(set(data_dict['labels'])))
        labels_dict = {i: label for i, label in enumerate(unique_labels)}
    print("Labels loaded:", labels_dict)
except FileNotFoundError:
    print("Error: data.pickle not found at:", DATA_PATH)
    exit()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

# ==============================
# MEDIAPIPE HANDS SETUP
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    max_num_hands=2
)

# ==============================
# PARAMETERS
# ==============================
CONFIDENCE_THRESHOLD = 0.82  # accuracy threshold
ALPHA = 0.65                 # smoothing for probabilities
BOX_ALPHA = 0.5              # smoothing for bounding box
SMOOTH_FRAMES = 3

# To keep per-hand smoothing data
hand_data = {}  # key: hand_idx, value: dict with 'smoothed_probs', 'last_pred', 'frame_count', 'smoothed_box'

print("\nStarting multi-hand inference – press 'q' to quit.\n")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_hands = results.multi_hand_landmarks if results.multi_hand_landmarks else []
    hand_labels = results.multi_handedness if results.multi_handedness else []

    # Track hand indices
    for idx, hand_landmarks in enumerate(detected_hands):
        handedness = hand_labels[idx].classification[0].label if hand_labels else 'Unknown'

        # Bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x1 = max(0, int(min(x_coords) * W) - 25)
        y1 = max(0, int(min(y_coords) * H) - 25)
        x2 = min(W, int(max(x_coords) * W) + 25)
        y2 = min(H, int(max(y_coords) * H) + 25)
        current_box = (x1, y1, x2, y2)

        # Initialize hand_data if new hand
        if idx not in hand_data:
            hand_data[idx] = {
                'smoothed_probs': None,
                'last_pred': "",
                'frame_count': 0,
                'smoothed_box': current_box
            }

        # Smooth bounding box
        prev_box = hand_data[idx]['smoothed_box']
        smoothed_box = tuple([
            int(BOX_ALPHA * c + (1 - BOX_ALPHA) * s)
            for c, s in zip(current_box, prev_box)
        ])
        hand_data[idx]['smoothed_box'] = smoothed_box

        # Normalize landmarks
        landmarks = hand_landmarks.landmark
        wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
        ref_x, ref_y = landmarks[12].x, landmarks[12].y
        hand_size = np.hypot(ref_x - wrist_x, ref_y - wrist_y) + 1e-6

        data_aux = []
        for lm in landmarks:
            dx = lm.x - wrist_x
            dy = lm.y - wrist_y
            if handedness == 'Right':
                dx = -dx
            data_aux.append(dx / hand_size)
            data_aux.append(dy / hand_size)

        # Predict probabilities
        current_probs = model.predict_proba([np.array(data_aux)])[0]

        # Smooth probabilities
        smoothed_probs = hand_data[idx]['smoothed_probs']
        if smoothed_probs is None:
            smoothed_probs = current_probs.copy()
        else:
            smoothed_probs = ALPHA * smoothed_probs + (1 - ALPHA) * current_probs
        hand_data[idx]['smoothed_probs'] = smoothed_probs

        # Get best prediction
        pred_index = np.argmax(smoothed_probs)
        confidence = smoothed_probs[pred_index]
        pred_label = labels_dict.get(pred_index, "?")

        # Display logic
        last_pred = hand_data[idx]['last_pred']
        frame_count = hand_data[idx]['frame_count']

        display_text = ""
        if confidence >= CONFIDENCE_THRESHOLD:
            if pred_label == last_pred:
                frame_count += 1
            else:
                frame_count = 1
                last_pred = pred_label

            if frame_count >= SMOOTH_FRAMES:
                display_text = pred_label
        else:
            frame_count = 0
            last_pred = ""

        hand_data[idx]['last_pred'] = last_pred
        hand_data[idx]['frame_count'] = frame_count

        # Draw bounding box and label
        x1, y1, x2, y2 = smoothed_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, handedness, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if display_text:
            text_size = 1.2
            text_thick = 2
            text_width, text_height = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, text_size, text_thick)[0]
            text_x = x2 - text_width - 10
            text_y = y1 + text_height + 10
            cv2.putText(frame, display_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, text_size, (0, 0, 0), text_thick + 2, cv2.LINE_AA)
            cv2.putText(frame, display_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, text_size, (60, 255, 140), text_thick, cv2.LINE_AA)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
