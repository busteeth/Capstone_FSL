import os
import cv2
import numpy as np
import random

# ==========================
# SETTINGS
# ==========================
DATA_DIR = './data_action'
VIDEO_DURATION = 3      # seconds per gesture
FPS = 30
AUGMENT_PER_VIDEO = 3
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

# ==========================
# AUGMENTATION FUNCTION
# ==========================
def augment_frame(frame):
    img = frame.copy()
    # Flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    # Brightness
    brightness = random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
    # Small rotation
    angle = random.uniform(-6, 6)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

# ==========================
# RECORD & AUGMENT FUNCTION
# ==========================
def record_class(class_name, num_videos=10):
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(class_dir) if f.endswith(".mp4")]
    counter = len(existing_files)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nRecording {num_videos} videos for '{class_name}'. Press Q to start each recording.")

    for vid in range(num_videos):
        # Wait for Q to start
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{class_name}: Press Q to START ({vid+1}/{num_videos})",
                        (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Start recording
        video_path = os.path.join(class_dir, f"{counter}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, FPS, (actual_w, actual_h))
        print(f"Recording video {vid+1}/{num_videos}: {video_path}")

        for frame_count in range(VIDEO_DURATION * FPS):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Recording... Press E to stop", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            out.write(frame)
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                print("Stopped early.")
                break

        out.release()
        counter += 1

        # ==========================
        # AUGMENTATION
        # ==========================
        cap_aug = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap_aug.read()
            if not ret:
                break
            frames.append(frame)
        cap_aug.release()

        for i in range(AUGMENT_PER_VIDEO):
            out_path = os.path.join(class_dir, f"{counter}.mp4")
            out = cv2.VideoWriter(out_path, fourcc, FPS, (actual_w, actual_h))
            for frame in frames:
                out.write(augment_frame(frame))
            out.release()
            print(f"Augmented video created: {out_path}")
            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAll videos for '{class_name}' recorded and augmented!\n")

# ==========================
# MAIN LOOP
# ==========================
os.makedirs(DATA_DIR, exist_ok=True)

while True:
    class_name = input("\nEnter gesture class (e.g., Yes, No, GoodMorning) or 'exit': ")
    if class_name.lower() == 'exit':
        break
    num_videos = input("Number of videos to record for this class (default 10): ")
    try:
        num_videos = int(num_videos)
    except:
        num_videos = 10
    record_class(class_name, num_videos=num_videos)

print("All dataset collection + augmentation complete!")
