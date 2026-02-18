import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV   # ← added for better probabilities

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)

if not os.path.exists(MODEL_DIR):
    print("Model directory not found:", MODEL_DIR)
    exit()


# List available dataset versions
datasets = sorted([f for f in os.listdir(MODEL_DIR)
                   if f.startswith('data.v') and f.endswith('.pickle')])

if not datasets:
    print("No data.v*.pickle files found in model/ folder.")
    print("Run create_dataset.py first.")
    exit()

print("\nAvailable dataset versions:")
for i, fname in enumerate(datasets, 1):
    path = os.path.join(MODEL_DIR, fname)
    size = os.path.getsize(path) / 1024 / 1024
    print(f"  {i:2d}) {fname}  ({size:.1f} MB)")

print("\nEnter number(s) to train on (e.g. 1  or  1 3 4  or 'all'):")
choice = input("> ").strip().lower()

if choice == 'all':
    selected = datasets
elif ' ' in choice:
    try:
        indices = [int(x)-1 for x in choice.split()]
        selected = [datasets[i] for i in indices if 0 <= i < len(datasets)]
    except:
        print("Invalid selection.")
        exit()
else:
    try:
        idx = int(choice) - 1
        selected = [datasets[idx]]
    except:
        print("Invalid selection.")
        exit()

if not selected:
    print("No valid datasets selected.")
    exit()

# ==============================
# LOAD & MERGE SELECTED DATASETS
# ==============================
all_data = []
all_labels = []

for fname in selected:
    path = os.path.join(MODEL_DIR, fname)
    print(f"Loading {fname} ...")
    with open(path, 'rb') as f:
        d = pickle.load(f)
        all_data.extend(d['data'])
        all_labels.extend(d['labels'])

print(f"Total samples loaded: {len(all_data)} from {len(selected)} version(s)")

data = np.asarray(all_data)
labels = np.asarray(all_labels)

# ==============================
# TRAIN
# ==============================
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(x_train, y_train)

# Calibrate probabilities (makes high confidence much more reliable)
print("Calibrating probabilities...")
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated_model.fit(x_test, y_test)

y_pred = calibrated_model.predict(x_test)
score = accuracy_score(y_pred, y_test)
print(f"\nAccuracy: {score*100:.2f}%  (test set)")

# Save (always to model.p)
MODEL_PATH = os.path.join(MODEL_DIR, 'model.p')
MODEL_BACKUP = os.path.join(MODEL_DIR, 'model_backup.p')

if os.path.exists(MODEL_PATH):
    os.replace(MODEL_PATH, MODEL_BACKUP)
    print("Old model → model_backup.p")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': calibrated_model}, f)

print(f"Model saved: {MODEL_PATH}")