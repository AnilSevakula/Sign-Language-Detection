import numpy as np
import os
from function import actions, no_sequences, sequence_length, DATA_PATH
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.callbacks import TensorBoard

# Create label mapping
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# Load sequences and labels
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            print(f"Loading: {path}")  # Debug print for file path
            res = np.load(path, allow_pickle=True)

            # Check if res has the expected shape
            if res.shape == (63,):  # Assuming you expect a 1D array of 63 elements
                window.append(res)
            else:
                print(f"Unexpected shape for {action}, sequence {sequence}, frame {frame_num}: {res.shape}")

        # Ensure window is of the expected length and shape
        if len(window) == sequence_length and all(w.shape == (63,) for w in window):
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Window length mismatch for action {action}, sequence {sequence}: {len(window)}")

# Convert sequences and labels to numpy arrays
X = np.array(sequences)  # Using default dtype to ensure consistent shapes
y = to_categorical(labels).astype(int)

# Reshape X to be compatible with CNN (samples, height, width, channels)
X = X.reshape(X.shape[0], sequence_length, 63, 1)  # Add a channel dimension

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the CNN model
model = Sequential()
model.add(Input(shape=(sequence_length, 63, 1)))  # Input layer for 2D CNN
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # 2D Convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # Another Conv layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
model.add(Flatten())  # Flatten the output from Conv layers
model.add(Dense(64, activation='relu'))  # Fully connected layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(len(actions), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

# Save the model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save('model.h5')
