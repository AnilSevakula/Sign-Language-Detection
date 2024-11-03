import os
import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)
directory = 'Img/'
labels = [chr(i) for i in range(65, 91)]  # Labels from A to Z
label_index = 0

# Ensure folders exist for each label
for label in labels:
    os.makedirs(os.path.join(directory, label), exist_ok=True)

print(f"Current label: {labels[label_index]}")

while True:
    # Capture frame-by-frame
    _, frame = cap.read()

    # Count images in the current label's folder
    count = len(os.listdir(os.path.join(directory, labels[label_index])))

    # Draw the rectangle and text
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.putText(frame, f"Current label: {labels[label_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
                2)
    cv2.putText(frame, "Press 'c' to capture, 'n' for next label, 'q' to quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    cv2.imshow("data", frame)

    # Extract region of interest
    roi = frame[40:400, 0:300]
    cv2.imshow("ROI", roi)

    # Handle key press events
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('c'):
        # Capture and save image for the current label
        img_path = os.path.join(directory, labels[label_index], f"{count}.png")
        cv2.imwrite(img_path, roi)
        print(f"Image saved: {img_path}")

    elif interrupt & 0xFF == ord('n'):
        # Move to the next label
        label_index = (label_index + 1) % len(labels)
        print(f"Current label: {labels[label_index]}")

    elif interrupt & 0xFF == ord('q'):
        # Quit the loop
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
