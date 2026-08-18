import cv2
import time
import math
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def main():
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            success, img = cap.read()
            attempt = 0
            while not success and attempt < 5:
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1
            if not success:
                print("Failed to read frame")
                break

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    lm = hand_landmarks.landmark
                    finger_fold_status = []

                    finger_tips = [8, 12, 16, 20]
                    for tip in finger_tips:
                        if lm[tip].x < lm[tip - 2].x:
                            finger_fold_status.append(True)
                        else:
                            finger_fold_status.append(False)

                    wrist = lm[0]
                    pinky_base = lm[17]
                    hand_size = distance(wrist, pinky_base)

                    thumb_tip = lm[4]
                    thumb_to_pinky = distance(thumb_tip, pinky_base)

                    thumb_folded = thumb_to_pinky < hand_size * 0.9
                    finger_fold_status.append(thumb_folded)

                    if all(finger_fold_status):
                        label = "Fist"
                    else:
                        label = "Open Hand"

                    cv2.putText(
                        img, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()