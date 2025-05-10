import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import time

model = load_model('sign_language_lstm.keras')
actions = np.load('actions.npy')
sequence = []
sequence_length = 30

predictions = deque(maxlen=30)
current_prediction = ''
last_added_letter = ''
word = ''

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - min(x_), lm.y - min(y_)])
        landmarks = np.array(landmarks)

        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        sequence.append(landmarks)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            pred = actions[np.argmax(res)]
            predictions.append(pred)

            most_common_pred, count = Counter(predictions).most_common(1)[0]
            if count > 15:
                current_prediction = most_common_pred
                if current_prediction != last_added_letter:
                    word += current_prediction
                    last_added_letter = current_prediction
    else:
        current_prediction = ''
        last_added_letter = ''

    cv2.putText(image, f"Letter: {current_prediction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(image, f"Word: {word}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    
    cv2.imshow('Webcam', image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

    if key == ord('c'):
        word = ''
        last_added_letter = ''
        current_prediction = ''
        sequence = []
        predictions.clear()

    if key == ord('s'):
        with open("recognized_words.txt", "a") as f:
            f.write(word + " ")
        print(f"Word saved: {word}")
        word = ''
        last_added_letter = ''
        current_prediction = ''
        sequence = []
        predictions.clear()

cap.release()
cv2.destroyAllWindows()
