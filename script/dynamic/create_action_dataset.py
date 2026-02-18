import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# ==========================
# PATHS
# ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, 'data_action')
MODEL_DIR = os.path.join(BASE_DIR, 'model_action')
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# VERSION INPUT
# ==========================
version_input = input("Enter version (e.g., 1.0) or press Enter for auto: ").strip()
if version_input == "":
    existing = [f for f in os.listdir(MODEL_DIR) if f.startswith("data_action.v") and f.endswith(".pickle")]
    versions = []
    for f in existing:
        try:
            v = f.replace("data_action.v", "").replace(".pickle", "")
            versions.append(float(v))
        except:
            pass
    version_input = str(round(max(versions)+0.1, 1)) if versions else "1.0"

SAVE_PATH = os.path.join(MODEL_DIR, f"data_action.v{version_input}.pickle")
print("Saving as:", SAVE_PATH)

# ==========================
# SETTINGS
# ==========================
SEQUENCE_LENGTH = 12
STEP = 3

# ==========================
# MEDIAPIPE
# ==========================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    max_num_hands=2  # TWO HANDS NOW
)

data = []
labels = []

classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print("Classes:", classes)

# ==========================
# PROCESS VIDEOS
# ==========================
for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)
    videos = [f for f in os.listdir(class_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for vid_name in videos:
        vid_path = os.path.join(class_path, vid_name)
        cap = cv2.VideoCapture(vid_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)
            current_frame = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Normalize
                    wrist_x, wrist_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                    ref_x, ref_y = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
                    hand_size = np.hypot(ref_x - wrist_x, ref_y - wrist_y) + 1e-6
                    hand_features = []
                    for lm in hand_landmarks.landmark:
                        dx = (lm.x - wrist_x) / hand_size
                        dy = (lm.y - wrist_y) / hand_size
                        hand_features.append(dx)
                        hand_features.append(dy)
                    current_frame.append(np.array(hand_features))

            # HANDLE TWO HANDS
            if len(current_frame) == 2:
                frame_features = np.concatenate(current_frame)  # 42+42 = 84
            elif len(current_frame) == 1:
                frame_features = np.concatenate([current_frame[0], np.zeros(42)])
            else:
                frame_features = np.zeros(84)

            frames.append(frame_features)

        cap.release()

        # ==========================
        # SLIDING WINDOW (SEQUENCES)
        # ==========================
        count = 0
        if len(frames) >= SEQUENCE_LENGTH:
            for i in range(0, len(frames) - SEQUENCE_LENGTH, STEP):
                sequence = frames[i:i+SEQUENCE_LENGTH]
                data.append(sequence)
                labels.append(cls)
                count += 1

        print(f"{vid_name}: {count} sequences added")

# ==========================
# SAVE DATA
# ==========================
with open(SAVE_PATH, "wb") as f:
    pickle.dump({
        "data": data,
        "labels": labels,
        "sequence_length": SEQUENCE_LENGTH,
        "step": STEP,
        "version": version_input
    }, f)

print("\nDONE")
print("Version:", version_input)
print("Total sequences:", len(data))
print("Saved at:", SAVE_PATH)
