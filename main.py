import time
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
def add_filter(frame, number):
    if number == 1:  # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if number == 2:  # Sepia
        matrix = np.array([[.13, .53, .27], [.17, .69, .35], [.19, .77, .39]])
        return np.clip(cv2.transform(frame, matrix), 0, 255).astype("uint8")
    if number == 3:  # Negative
        return 255 - frame
    if number == 4:  # Blur
        return cv2.GaussianBlur(frame, (21, 21), 0)
    return frame
camera = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
filter_number = 0
last_gesture = 0
print("SPACE = take photo | 0-4 = filters | Q = quit")
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    result = hand_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        points = result.multi_hand_landmarks[0].landmark
        thumb = points[4]
        palm = np.hypot((points[0].x - points[9].x) * width,
                        (points[0].y - points[9].y) * height)
        palm = max(palm, 1)
        for tip, new_filter in [(12, 1), (16, 2), (20, 3)]:
            distance = np.hypot((thumb.x - points[tip].x) * width,
                                (thumb.y - points[tip].y) * height)
            if distance < palm * .35 and time.time() - last_gesture > 1:
                filter_number = new_filter
                last_gesture = time.time()
                break
        index = points[8]
        distance = np.hypot((thumb.x - index.x) * width,
                            (thumb.y - index.y) * height)
        if distance < palm * .35 and time.time() - last_gesture > 1:
            name = f"photo_{datetime.now():%H%M%S}.jpg"
            cv2.imwrite(name, add_filter(frame, filter_number))
            print("Saved", name)
            last_gesture = time.time()
    preview = add_filter(frame, filter_number)
    cv2.putText(preview, f"Filter {filter_number} | SPACE: photo | 0-4: filter | Q: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    cv2.imshow("Gesture Photo", preview)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if ord("0") <= key <= ord("4"):
        filter_number = key - ord("0")
    if key == ord(" "):
        name = f"photo_{datetime.now():%H%M%S}.jpg"
        cv2.imwrite(name, add_filter(frame, filter_number))
        print("Saved", name)
camera.release()
hand_detector.close()
cv2.destroyAllWindows()
