import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join('MP_Data')

actions = np.array([chr(i) for i in range(ord('A'), ord('Z')+1)])
sequence_length = 30

sequences, labels = [], []
for action_idx, action in enumerate(actions):
    for sequence in os.listdir(os.path.join(DATA_PATH, action)):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, sequence, f"{frame_num}.npy"))
                window.append(res)
            except:
                continue
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(action_idx)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X = X / np.max(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len(actions), activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_categorical_accuracy', save_best_only=True)

history = model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, checkpoint],
    batch_size=32
)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save('sign_language_lstm.keras')
np.save('actions.npy', actions)

y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=actions))