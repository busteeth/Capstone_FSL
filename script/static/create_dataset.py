import os
import pickle
import mediapipe as mp
import cv2
from pathlib import Path

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("MODEL_DIR:", MODEL_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ==============================
# Ask for version
# ==============================
print("\n=== Dataset Creation ===")
version_input = input("Enter version (e.g. 1.0, 2.3) or press Enter to auto-increment: ").strip()

if version_input:
    version = version_input
else:
    existing = [f for f in os.listdir(MODEL_DIR) if f.startswith('data.v') and f.endswith('.pickle')]
    versions = []
    for f in existing:
        try:
            v = f.replace('data.v', '').replace('.pickle', '')
            major, minor = map(int, v.split('.'))
            versions.append((major, minor))
        except:
            pass
    if versions:
        major, minor = max(versions, key=lambda x: (x[0], x[1]))
        version = f"{major}.{minor + 1}"
    else:
        version = "1.0"

print(f"→ Using version: {version}")

DATA_PICKLE_PATH = os.path.join(MODEL_DIR, f'data.v{version}.pickle')

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ==============================
# LOAD EXISTING DATA (if this version already exists)
# ==============================
data = []
labels = []

if os.path.exists(DATA_PICKLE_PATH):
    print(f"Version {version} already exists. Loading and appending new samples...")
    with open(DATA_PICKLE_PATH, 'rb') as f:
        existing = pickle.load(f)
        data = existing.get('data', [])
        labels = existing.get('labels', [])
else:
    print(f"Creating new dataset version {version}...")

# ==============================
# GET CLASSES & COUNT
# ==============================
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
if not classes:
    print("No class subfolders found in 'data' folder!")
    exit()

total_images = sum(len(os.listdir(os.path.join(DATA_DIR, c))) for c in classes)
if total_images == 0:
    print("No images found.")
    exit()

# ==============================
# PROCESS (NEW: wrist-centered + mirror right hands)
# ==============================
new_count = 0
processed_count = 0

print(f"\nProcessing {total_images} images from {len(classes)} classes...\n")

for dir_ in classes:
    class_path = os.path.join(DATA_DIR, dir_)
    img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    for idx, img_name in enumerate(img_files, 1):
        processed_count += 1
        img_full_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_full_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = results.multi_handedness[hand_idx].classification[0].label if results.multi_handedness else 'Unknown'

                landmarks = hand_landmarks.landmark
                wrist_x = landmarks[0].x
                wrist_y = landmarks[0].y

                # Hand size: wrist → middle finger tip (landmark 12)
                ref_x = landmarks[12].x
                ref_y = landmarks[12].y
                hand_size = ((ref_x - wrist_x)**2 + (ref_y - wrist_y)**2)**0.5 + 1e-6

                data_aux = []
                for lm in landmarks:
                    dx = lm.x - wrist_x
                    dy = lm.y - wrist_y
                    if handedness == 'Right':
                        dx = -dx  # mirror right hand to look like left
                    data_aux.append(dx / hand_size)
                    data_aux.append(dy / hand_size)

                if len(data_aux) == 42 and data_aux not in data:
                    data.append(data_aux)
                    labels.append(dir_)
                    new_count += 1

        if processed_count % 10 == 0 or processed_count == total_images:
            print(f"Processed {processed_count}/{total_images} images...")

# ==============================
# SAVE
# ==============================
with open(DATA_PICKLE_PATH, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\nDone!")
print(f"Version:      v{version}")
print(f"New samples:  {new_count}")
print(f"Total samples:{len(data)}")
print(f"Saved:        {DATA_PICKLE_PATH}\n")