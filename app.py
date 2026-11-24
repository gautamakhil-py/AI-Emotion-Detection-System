from fer import FER
import cv2
import csv
from datetime import datetime

# Emotion Detector
detector = FER()

# Camera
cap = cv2.VideoCapture(0)

# Camera Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera open nahi hua!")
    exit()

# Large Window
cv2.namedWindow("AI Emotion Detection System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Emotion Detection System", 1200, 700)

paused = False

print("=" * 50)
print("AI Emotion Detection System Started")
print("SPACE = Pause/Resume")
print("S = Save Screenshot")
print("Q = Quit")
print("=" * 50)

while True:

    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("Frame read failed!")
            break

        results = detector.detect_emotions(frame)

        for result in results:

            x, y, w, h = result["box"]

            emotions = result["emotions"]

            dominant_emotion = max(
                emotions,
                key=emotions.get
            )

            confidence = emotions[dominant_emotion]

            # Emotion Colors
            if dominant_emotion == "happy":
                color = (0, 255, 0)

            elif dominant_emotion == "sad":
                color = (255, 0, 0)

            elif dominant_emotion == "angry":
                color = (0, 0, 255)

            else:
                color = (0, 255, 255)

            # Rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

            # Emotion Text
            text = (
                f"{dominant_emotion.upper()} "
                f"{confidence*100:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Save Log
            with open(
                "emotion_log.csv",
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    datetime.now(),
                    dominant_emotion,
                    round(confidence * 100, 2)
                ])

    cv2.imshow(
        "AI Emotion Detection System",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Pause / Resume
    if key == 32:
        paused = not paused

        if paused:
            print("Paused")
        else:
            print("Resumed")

    # Screenshot
    elif key == ord('s'):

        filename = (
            "screenshot_" +
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            ) +
            ".jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Screenshot Saved: {filename}"
        )

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
