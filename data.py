import os
import cv2
import numpy as np
import mediapipe as mp

DATA_PATH = os.path.join('MP_Data_1')  
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])  
no_sequences = 30 
sequence_length = 30  

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for action in actions:
    for sequence in range(no_sequences):
        while True:
            ret, frame = cap.read()
            text = f'Get ready: {action} sample {sequence} - Press "Q" to start!'
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        print(f'[INFO] Recording {action} sample {sequence}')
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            landmarks = np.zeros(42)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - min(x_), lm.y - min(y_)])
                landmarks = np.array(landmarks)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(image, f'{action} | Seq {sequence} | Frame {frame_num}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Webcam', image)

            if len(landmarks) == 42:
                np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), landmarks)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
