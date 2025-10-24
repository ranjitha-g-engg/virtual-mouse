import cv2
import mediapipe as mp
import csv
import os

# --- Configuration ---
DATA_FILE = 'gestures.csv'
GESTURE_LABELS = {
    'p': 'pointing',
    'c': 'pinching',
    'f': 'neutral',
}

# --- Create CSV header ---
header = ['label']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

# Check if file exists, if not create it with a header
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

print("Starting data collection. A window will appear showing your webcam.")
print("Position your hand clearly in the frame.")
print("Press a key to start labeling the current gesture:")
for key, gesture in GESTURE_LABELS.items():
    print(f"  - Press and HOLD '{key}' to save data for {gesture}")
print("\nPress 'q' to quit.")

cap = cv2.VideoCapture(0) # Use 0 for the default laptop webcam
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1) # Flip for selfie view
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        annotated_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('Data Collection - Press keys to label', annotated_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        if chr(key) in GESTURE_LABELS:
            gesture_label = GESTURE_LABELS[chr(key)]
            
            # Save data
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                row = [gesture_label]
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                
                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                print(f"Saved data point for: {gesture_label}")

cap.release()
cv2.destroyAllWindows()
print(f"Data collection complete. Data saved to '{DATA_FILE}'.")