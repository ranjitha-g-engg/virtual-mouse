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
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Screen and Mouse Control Setup ---
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING_FACTOR = 4 # You can tune this (lower = faster, higher = smoother)
plocx, plocy = 0, 0
clocx, clocy = 0, 0
is_pinching = False

print("Final Virtual Mouse is running. Put your hand in the frame. Press 'q' to quit.")

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image.flags.writeable = False
        results = hands.process(image_rgb)
        image.flags.writeable = True
        
        display_text = "No Hand Detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Predict the gesture
                landmarks = hand_landmarks.landmark
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                
                X = np.array(row).reshape(1, -1)
                predicted_gesture = model.predict(X)[0]
                probabilities = model.predict_proba(X)
                confidence = np.max(probabilities)
                
                display_text = f"{predicted_gesture} ({confidence*100:.2f}%)"

                # 2. Get the coordinates of the index fingertip
                index_fingertip = landmarks[8]
                x, y = index_fingertip.x, index_fingertip.y

                # 3. Perform actions based on gesture and confidence
                if predicted_gesture == 'pointing':
                    screen_x = np.interp(x, (0.1, 0.5), (0, SCREEN_WIDTH))
                    screen_y = np.interp(y, (0.1, 0.5), (0, SCREEN_HEIGHT))

                    clocx = plocx + (screen_x - plocx) / SMOOTHING_FACTOR
                    clocy = plocy + (screen_y - plocy) / SMOOTHING_FACTOR
                    
                    pyautogui.moveTo(clocx, clocy)
                    plocx, plocy = clocx, clocy
                    is_pinching = False
                
                # --- THIS IS THE UPDATED CLICK LOGIC ---
                elif predicted_gesture == 'pinching':
                    # Only click if confidence is high and we haven't clicked already
                    if not is_pinching and confidence > 0.75: # 85% confidence threshold
                        pyautogui.click()
                        print(f"Clicked with {confidence*100:.2f}% confidence.")
                        is_pinching = True # Set flag to prevent repeated clicks
                
                elif predicted_gesture == 'neutral':
                    is_pinching = False # Reset the click flag
                
                # Draw landmarks for visualization
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the predicted gesture on the screen
        cv2.rectangle(image, (30, 30), (400, 80), (0, 0, 0), -1)
        cv2.putText(image, display_text, (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Virtual Mouse', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()