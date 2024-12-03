import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import datetime  # Import datetime module to check the time

# Initialize Video Capture and Models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "C:/Users/hudda/OneDrive/Documents/Desktop/sign language detection/keras_model.h5",
    "C:/Users/hudda/OneDrive/Documents/Desktop/sign language detection/labels.txt"
)

# Constants
offset = 20
imgSize = 300
labels = ["Hello", "Yes", "I love you", "Thank you", "Ok"]

while True:
    # Get the current time
    current_time = datetime.datetime.now()
    current_hour = current_time.hour

    # Check if the current time is between 6 PM (18:00) and 10 PM (22:00)
    if current_hour >= 18 and current_hour < 22:
        success, img = cap.read()

        # Ensure the frame is captured correctly
        if not success:
            print("Failed to capture video frame.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Validate and adjust bounding box
            height, width, _ = img.shape
            y1, y2 = max(0, y - offset), min(y + h + offset, height)
            x1, x2 = max(0, x - offset), min(x + w + offset, width)

            # Crop and process the hand region
            imgCrop = img[y1:y2, x1:x2]

            # Skip empty crops
            if imgCrop.size == 0:
                print("Empty crop detected. Skipping frame.")
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                # Height is greater
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # Width is greater
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Make prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {labels[index]} ({prediction})")

            # Display results on the frame
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 200, y - offset), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        # Display main image
        cv2.imshow("Image", imgOutput)

    else:
        print("Model is inactive. It runs only between 6 PM and 10 PM.")

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
