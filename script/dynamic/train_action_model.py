import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================
# PATHS
# ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_DIR = os.path.join(BASE_DIR, "model_action")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# PICK DATASET
# ==========================
datasets = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pickle")]
if not datasets:
    print("No datasets found in", MODEL_DIR)
    exit()

print("Available datasets:")
for i, f in enumerate(datasets, 1):
    print(f"{i}. {f}")

choice = input("Enter the number of the dataset you want to use: ").strip()
try:
    choice_idx = int(choice) - 1
    DATA_PATH = os.path.join(MODEL_DIR, datasets[choice_idx])
except:
    print("Invalid choice.")
    exit()

print("Using dataset:", DATA_PATH)

# ==========================
# LOAD DATA
# ==========================
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"], dtype=np.float32)
y = dataset["labels"]

unique_labels = sorted(list(set(y)))
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
y_idx = np.array([label_to_idx[label] for label in y])

print(f"Samples: {len(X)}")
print(f"Classes: {unique_labels}")

# ==========================
# MODEL
# ==========================
input_size = X.shape[2]  # automatically 42*2=84 if 2-hand
hidden_size = 256
num_layers = 3
num_classes = len(unique_labels)

class SignLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = SignLSTM()

# ==========================
# TRAIN
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50  # longer training

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_idx, dtype=torch.long)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ==========================
# SAVE MODEL
# ==========================
MODEL_PATH = os.path.join(MODEL_DIR, "model_action.p")
torch.save({
    "model_state": model.state_dict(),
    "unique_labels": unique_labels,
    "input_size": input_size
}, MODEL_PATH)

print("\nTraining complete!")
print("Model saved to:", MODEL_PATH)
