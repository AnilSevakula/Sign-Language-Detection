import cv2
import numpy as np
from keras.models import model_from_json
from function import actions, sequence_length, extract_keypoints, mediapipe_detection
import mediapipe as mp

# Load the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Color list for action visualization
colors = [(245, 117, 16)] * len(actions)

def prob_viz(res, actions, input_frame, colors, threshold):
    """Visualize action probabilities on the frame."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

# Initialize detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(1)  # Use 0 for the built-in webcam or replace with video path

# Initialize Mediapipe Hands model
with mp.solutions.hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Define and highlight region of interest (ROI) on the frame
        crop_frame = frame[40:400, 0:300]  # Adjust the cropping dimensions if necessary
        cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

        # Process the cropped frame with MediaPipe
        image, results = mediapipe_detection(crop_frame, hands)
        
        # Extract keypoints if landmarks are detected
        keypoints = extract_keypoints(results)
        if keypoints is not None:
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]  # Maintain a fixed length

            # Predict only if we have the required sequence length
            if len(sequence) == sequence_length:
                # Make a prediction
                input_data = np.expand_dims(sequence, axis=0)
                res = model.predict(input_data)[0]
                
                # Add prediction to the history
                predictions.append(np.argmax(res))

                # Check for confident prediction and consistent action
                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                    action_name = actions[np.argmax(res)]
                    action_accuracy = f"{res[np.argmax(res)] * 100:.2f}%"

                    # Update only if it's a new action or if `sentence` is empty
                    if len(sentence) == 0 or action_name != sentence[-1]:
                        sentence = [action_name]  # Update with new action
                        accuracy = [action_accuracy]  # Update accuracy

                # Visualization of probabilities
                frame = prob_viz(res, actions, frame, colors, threshold)

        # Display output text
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, f"Output: {sentence[-1] if sentence else ''} ({accuracy[-1] if accuracy else ''})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Real-Time Detection', frame)

        # Refresh the display on pressing 'c' and clear previous actions
        if cv2.waitKey(10) & 0xFF == ord('c'):
            sequence = []  # Clear the sequence
            sentence = []   # Clear previous sentences
            accuracy = []   # Clear previous accuracy data
            predictions = []  # Clear previous predictions

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
