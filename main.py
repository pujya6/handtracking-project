import cv2
import time
import math
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient


# OSC client
OSC_IP = "127.0.0.1"
OSC_PORT = 7000

osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)


# MediaPipe setup
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


# Count fingers
def count_fingers(lm, hand_label):

    fingers = []

    # Index finger
    if lm[8].y < lm[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle finger
    if lm[12].y < lm[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Ring finger
    if lm[16].y < lm[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Pinky finger
    if lm[20].y < lm[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Thumb
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


# Main function
def main():

    previous_time = 0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:

            # Read webcam
            success, img = cap.read()

            attempt = 0

            while not success and attempt < 5:
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1

            if not success:
                print("Failed to read frame")
                break


            # Mirror webcam
            img = cv2.flip(img, 1)


            # Convert BGR to RGB
            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )


            # Process image
            results = hands.process(rgb)


            # Process detected hands
            if results.multi_hand_landmarks:

                # --------------------------------
                # DRAW AND DISPLAY ALL DETECTED HANDS
                # --------------------------------

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )


                    # Get landmarks
                    lm = hand_landmarks.landmark


                    # Identify hand
                    hand_label = handedness.classification[0].label


                    # Count fingers
                    finger_count = count_fingers(
                        lm,
                        hand_label
                    )


                    # Determine gesture
                    if finger_count == 0:
                        gesture = "Fist"

                    elif finger_count == 5:
                        gesture = "Open Hand"

                    else:
                        gesture = "Gesture"


                    # Find position for text
                    wrist = lm[0]

                    h, w, _ = img.shape

                    text_x = int(wrist.x * w)
                    text_y = int(wrist.y * h) - 30


                    # Keep text inside screen
                    text_x = max(10, text_x)
                    text_y = max(40, text_y)


                    # Display hand information
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


                # --------------------------------
                # FIRST HAND DATA
                # --------------------------------

                lm1 = results.multi_hand_landmarks[0].landmark

                hand1_label = (
                    results.multi_handedness[0]
                    .classification[0]
                    .label
                )

                hand1_fingers = count_fingers(
                    lm1,
                    hand1_label
                )

                hand1_x = lm1[9].x
                hand1_y = lm1[9].y


                # IMPORTANT:
                # These addresses stay the same so
                # your existing TouchDesigner setup
                # continues to work.

                osc_client.send_message(
                    "/hand_x",
                    hand1_x
                )

                osc_client.send_message(
                    "/hand_y",
                    hand1_y
                )

                osc_client.send_message(
                    "/fingers",
                    hand1_fingers
                )


                # --------------------------------
                # SECOND HAND DATA
                # --------------------------------

                if len(results.multi_hand_landmarks) > 1:

                    lm2 = results.multi_hand_landmarks[1].landmark

                    hand2_label = (
                        results.multi_handedness[1]
                        .classification[0]
                        .label
                    )

                    hand2_fingers = count_fingers(
                        lm2,
                        hand2_label
                    )

                    hand2_x = lm2[9].x
                    hand2_y = lm2[9].y


                    # Send second hand data
                    osc_client.send_message(
                        "/hand2_x",
                        hand2_x
                    )

                    osc_client.send_message(
                        "/hand2_y",
                        hand2_y
                    )

                    osc_client.send_message(
                        "/hand2_fingers",
                        hand2_fingers
                    )


                    # --------------------------------
                    # DISTANCE BETWEEN TWO HANDS
                    # --------------------------------

                    dx = hand1_x - hand2_x
                    dy = hand1_y - hand2_y

                    hands_distance = math.sqrt(
                        dx**2 + dy**2
                    )

                    osc_client.send_message(
                        "/hands_distance",
                        hands_distance
                    )


                else:

                    # --------------------------------
                    # NO SECOND HAND DETECTED
                    # --------------------------------

                    osc_client.send_message(
                        "/hand2_x",
                        0
                    )

                    osc_client.send_message(
                        "/hand2_y",
                        0
                    )

                    osc_client.send_message(
                        "/hand2_fingers",
                        0
                    )

                    osc_client.send_message(
                        "/hands_distance",
                        0
                    )


            # --------------------------------
            # CALCULATE FPS
            # --------------------------------

            current_time = time.time()

            fps = (
                1 / (current_time - previous_time)
                if previous_time != 0
                else 0
            )

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
            cv2.imshow(
                "Hand Tracking",
                img
            )


            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    # Release camera
    cap.release()

    cv2.destroyAllWindows()


# Run program
if __name__ == "__main__":
    main()