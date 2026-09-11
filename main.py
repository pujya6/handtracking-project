import cv2
import time
import math
import mediapipe as mp

#mediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

# Calculate distance between two landmarks
def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )

#count fingers
def count_fingers(lm, hand_label):

    fingers = []

    #index finger
    if lm[8].y < lm[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    #middle finger
    if lm[12].y < lm[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    #ring finger
    if lm[16].y < lm[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    #pinky finger
    if lm[20].y < lm[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    #thumb
    if hand_label == "Right":
        if lm[4].x < lm[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    else:
        if lm[4].x > lm[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

#main function
def main():

    previous_time = 0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:

            #read webcam
            success, img = cap.read()

            attempt = 0

            while not success and attempt < 5:
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1

            if not success:
                print("Failed to read frame")
                break

            #mirror webcam
            img = cv2.flip(img, 1)

            #convert BGR to RGB
            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            #process image
            results = hands.process(rgb)

            #process detected hands
            if results.multi_hand_landmarks:

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    #draw landmarks
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    #get landmarks
                    lm = hand_landmarks.landmark

                    # Identify hand
                    hand_label = handedness.classification[0].label

                    #count fingers
                    finger_count = count_fingers(
                        lm,
                        hand_label
                    )

                    #determine gesture
                    if finger_count == 0:
                        gesture = "Fist"

                    elif finger_count == 5:
                        gesture = "Open Hand"

                    else:
                        gesture = "Gesture"

                    #find position for text
                    wrist = lm[0]

                    h, w, _ = img.shape

                    text_x = int(wrist.x * w)
                    text_y = int(wrist.y * h) - 30

                    #keep text inside screen
                    text_x = max(10, text_x)
                    text_y = max(40, text_y)

                    #display information
                    cv2.putText(
                        img,
                        f"{hand_label} Hand",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        img,
                        f"Fingers: {finger_count}",
                        (text_x, text_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        img,
                        gesture,
                        (text_x, text_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            # Calculate FPS
            current_time = time.time()

            fps = 1 / (current_time - previous_time) \
                if previous_time != 0 else 0

            previous_time = current_time

            cv2.putText(
                img,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            # Display webcam
            cv2.imshow("Hand Tracking", img)

            #press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    #release camera
    cap.release()
    cv2.destroyAllWindows()

#run program
if __name__ == "__main__":
    main()