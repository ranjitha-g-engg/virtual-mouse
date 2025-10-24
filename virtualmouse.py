import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyautogui

# --- Load the Trained Model ---
MODEL_FILE = 'gesture_model.pkl'
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

# --- Webcam and Hand Tracking Setup ---
cap = cv2.VideoCapture(0) # Use 0 for the default laptop webcam
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Screen and Mouse Control Setup ---
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING_FACTOR = 4 # Higher value = smoother, but more lag
plocx, plocy = 0, 0 # Previous location
clocx, clocy = 0, 0 # Current location
is_pinching = False

print("Virtual Mouse is running. Put your hand in the frame. Press 'q' to quit.")

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Predict the gesture
                landmarks = hand_landmarks.landmark
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                
                X = np.array(row).reshape(1, -1)
                predicted_gesture = model.predict(X)[0]

                # 2. Get the coordinates of the index fingertip (landmark #8)
                index_fingertip = landmarks[8]
                x, y = index_fingertip.x, index_fingertip.y

                # 3. Perform actions
                if predicted_gesture == 'pointing':
                    # Map hand coordinates to screen coordinates
                    screen_x = np.interp(x, (0.05, 0.95), (0, SCREEN_WIDTH))
                    screen_y = np.interp(y, (0.05, 0.95), (0, SCREEN_HEIGHT))

                    # Smooth the movement
                    clocx = plocx + (screen_x - plocx) / SMOOTHING_FACTOR
                    clocy = plocy + (screen_y - plocy) / SMOOTHING_FACTOR
                    
                    pyautogui.moveTo(clocx, clocy)
                    plocx, plocy = clocx, clocy
                    is_pinching = False
                
                elif predicted_gesture == 'pinching':
                    if not is_pinching:
                        pyautogui.click()
                        print("Click!")
                        is_pinching = True
                
                elif predicted_gesture == 'neutral':
                    is_pinching = False
                
                # Draw the predicted gesture on the screen
                cv2.putText(image, predicted_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Virtual Mouse', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()