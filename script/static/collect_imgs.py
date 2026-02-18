import os
import cv2

# Settings
DATA_DIR = './data'
dataset_size = 500

# Create main folder
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
while True:

    # Ask folder name FIRST
    class_name = input("\nEnter folder name or type 'exit': ")

    if class_name.lower() == "exit":
        print("Exiting program...")
        break

    class_dir = os.path.join(DATA_DIR, class_name)

    # Create folder if not exist
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        print(f"Folder '{class_name}' created.")
    else:
        print(f"Folder '{class_name}' already exists. Adding more images...")

    # Open camera ONLY after folder name
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        break

    print(f"Press 'Q' to start capturing for '{class_name}'")
    print("Press 'E' to cancel\n")

    # Wait until Q pressed
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(frame,
                    f"{class_name}: Press Q to start",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('e'):
            cap.release()
            cv2.destroyAllWindows()
            continue

    # Start capturing
    existing_files = len(os.listdir(class_dir))
    counter = existing_files

    print(f"Capturing {dataset_size} images for '{class_name}'...")

    while counter < existing_files + dataset_size:

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(frame,
                    f"{class_name} Image {counter-existing_files+1}/{dataset_size}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.imshow("Camera", frame)

        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)

        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    print(f"Done capturing for '{class_name}'")

    cap.release()
    cv2.destroyAllWindows()

print("Dataset collection complete!")
