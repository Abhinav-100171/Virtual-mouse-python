import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
import urllib.request


# =====================================================
# SETTINGS
# =====================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

screen_width, screen_height = pyautogui.size()

MODEL_FILE = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

# Cursor smoothing
SMOOTHING = 0.35

# Safe camera area
CAMERA_MARGIN = 0.08

# Pinch distance
PINCH_THRESHOLD = 0.045

# Timing
DOUBLE_CLICK_TIME = 0.35
HOLD_TIME = 0.7
GESTURE_COOLDOWN = 1.0


# =====================================================
# DOWNLOAD MEDIAPIPE MODEL
# =====================================================

if not os.path.exists(MODEL_FILE):

    print("Downloading MediaPipe hand model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_FILE
    )

    print("Hand model downloaded successfully.")


# =====================================================
# MEDIAPIPE
# =====================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# =====================================================
# VARIABLES
# =====================================================

previous_x = screen_width // 2
previous_y = screen_height // 2

cursor_speed = 1.0

mouse_holding = False

mouse_paused = False

pinch_start_time = None

last_click_time = 0

last_lock_time= 0

last_media_time = 0

last_screenshot_time = 0

last_window_time = 0

last_volume_time = 0

last_zoom_time = 0

last_action_time = 0

previous_pinch = False

previous_two_finger_distance = None

previous_swipe_x = None

fist_start_time = None

running = True
pyautogui.FAILSAFE = False


# =====================================================
# FUNCTIONS
# =====================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def count_fingers(hand):

    count = 0

    # Index
    if hand[8].y < hand[6].y:
        count += 1

    # Middle
    if hand[12].y < hand[10].y:
        count += 1

    # Ring
    if hand[16].y < hand[14].y:
        count += 1

    # Pinky
    if hand[20].y < hand[18].y:
        count += 1

    return count


def show_text(frame, message, y):

    cv2.putText(
        frame,
        message,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


# =====================================================
# MAIN PROGRAM
# =====================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while running:

        success, frame = cap.read()

        if not success:

            print("Could not read camera.")

            break


        # Mirror camera
        frame = cv2.flip(frame, 1)


        # Convert to RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # Detect hand
        result = landmarker.detect(mp_image)


        # =================================================
        # HAND DETECTED
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ---------------------------------------------
            # FINGER POINTS
            # ---------------------------------------------

            thumb = hand[4]

            index = hand[8]

            middle = hand[12]

            ring = hand[16]

            pinky = hand[20]


            # ---------------------------------------------
            # COUNT FINGERS
            # ---------------------------------------------

            fingers = count_fingers(hand)


            # ---------------------------------------------
            # PINCH
            # ---------------------------------------------

            pinch_distance = distance(
                thumb,
                index
            )

            pinch = (
                pinch_distance < PINCH_THRESHOLD
            )


            # =================================================
            # NORMAL MOUSE
            # =================================================

            if not mouse_paused:


                # =============================================
                # CURSOR MOVEMENT
                # =============================================

                if not pinch:

                    # Ignore extreme camera edges
                    margin = CAMERA_MARGIN

                    safe_x = max(
                        margin,
                        min(
                            1 - margin,
                            index.x
                        )
                    )

                    safe_y = max(
                        margin,
                        min(
                            1 - margin,
                            index.y
                        )
                    )


                    # Convert camera position to screen position
                    target_x = int(
                        (
                            (safe_x - margin)
                            / (1 - 2 * margin)
                        )
                        * screen_width
                    )

                    target_y = int(
                        (
                            (safe_y - margin)
                            / (1 - 2 * margin)
                        )
                        * screen_height
                    )


                    # Smooth movement
                    new_x = (
                        previous_x
                        +
                        (
                            target_x
                            - previous_x
                        )
                        * SMOOTHING
                        * cursor_speed
                    )

                    new_y = (
                        previous_y
                        +
                        (
                            target_y
                            - previous_y
                        )
                        * SMOOTHING
                        * cursor_speed
                    )


                    # Extra safety
                    new_x = max(
                        5,
                        min(
                            screen_width - 5,
                            int(new_x)
                        )
                    )

                    new_y = max(
                        5,
                        min(
                            screen_height - 5,
                            int(new_y)
                        )
                    )


                    pyautogui.moveTo(
                        new_x,
                        new_y
                    )


                    previous_x = new_x
                    previous_y = new_y


                # =============================================
                # CLICK / DOUBLE CLICK / HOLD / DRAG
                # =============================================

                if pinch:

                    if pinch_start_time is None:

                        pinch_start_time = time.time()


                    pinch_duration = (
                        time.time()
                        - pinch_start_time
                    )


                    # Hold and drag
                    if (
                        pinch_duration >= HOLD_TIME
                        and
                        not mouse_holding
                    ):

                        pyautogui.mouseDown()

                        mouse_holding = True

                        show_text(
                            frame,
                            "CLICK & HOLD / DRAG",
                            40
                        )


                else:

                    # Pinch released
                    if pinch_start_time is not None:

                        duration = (
                            time.time()
                            - pinch_start_time
                        )


                        # If it wasn't a hold,
                        # treat as click
                        if (
                            duration < HOLD_TIME
                            and
                            not mouse_holding
                        ):

                            current_time = time.time()


                            # Double click
                            if (
                                current_time
                                - last_click_time
                                < DOUBLE_CLICK_TIME
                            ):

                                pyautogui.doubleClick()

                                show_text(
                                    frame,
                                    "DOUBLE CLICK",
                                    40
                                )

                                last_click_time = 0


                            else:

                                pyautogui.click()

                                show_text(
                                    frame,
                                    "LEFT CLICK",
                                    40
                                )

                                last_click_time = (
                                    current_time
                                )


                    # Release drag
                    if mouse_holding:

                        pyautogui.mouseUp()

                        mouse_holding = False

                        show_text(
                            frame,
                            "DROP",
                            40
                        )


                    pinch_start_time = None


                # =============================================
                # RIGHT CLICK
                # =============================================

                # Thumb + middle finger
                thumb_middle_distance = distance(
                    thumb,
                    middle
                )

                right_pinch = (
                    thumb_middle_distance
                    < PINCH_THRESHOLD
                )


                if (
                    right_pinch
                    and
                    not pinch
                    and
                    time.time()
                    - last_click_time
                    > 0.5
                ):

                    pyautogui.rightClick()

                    show_text(
                        frame,
                        "RIGHT CLICK",
                        40
                    )

                    last_click_time = time.time()


                # =============================================
                # TWO FINGERS = SCROLL
                # =============================================

                if fingers == 2 and not pinch:

                    if index.y < 0.35:

                        pyautogui.scroll(3)

                        show_text(
                            frame,
                            "SCROLL UP",
                            40
                        )

                    elif index.y > 0.65:

                        pyautogui.scroll(-3)

                        show_text(
                            frame,
                            "SCROLL DOWN",
                            40
                        )


                # =============================================
                # ZOOM
                # =============================================

                if fingers == 2:

                    finger_distance = distance(
                        index,
                        middle
                    )


                    if (
                        previous_two_finger_distance
                        is not None
                    ):

                        change = (
                            finger_distance
                            - previous_two_finger_distance
                        )


                        if (
                            change > 0.04
                            and
                            time.time()
                            - last_zoom_time
                            > 0.6
                        ):

                            pyautogui.hotkey(
                                "ctrl",
                                "+"
                            )

                            show_text(
                                frame,
                                "ZOOM IN",
                                40
                            )

                            last_zoom_time = (
                                time.time()
                            )


                        elif (
                            change < -0.04
                            and
                            time.time()
                            - last_zoom_time
                            > 0.6
                        ):

                            pyautogui.hotkey(
                                "ctrl",
                                "-"
                            )

                            show_text(
                                frame,
                                "ZOOM OUT",
                                40
                            )

                            last_zoom_time = (
                                time.time()
                            )


                    previous_two_finger_distance = (
                        finger_distance
                    )

                else:

                    previous_two_finger_distance = None


                # =============================================
                # THUMBS UP = VOLUME UP
                # =============================================

                if (
                    fingers == 0
                    and
                    thumb.y < hand[3].y
                ):

                    if (
                        time.time()
                        - last_volume_time
                        > 0.5
                    ):

                        pyautogui.press(
                            "volumeup"
                        )

                        show_text(
                            frame,
                            "VOLUME UP",
                            40
                        )

                        last_volume_time = (
                            time.time()
                        )


                # =============================================
                # THUMBS DOWN = VOLUME DOWN
                # =============================================

                if (
                    fingers == 0
                    and
                    thumb.y > hand[3].y
                ):

                    if (
                        time.time()
                        - last_volume_time
                        > 0.5
                    ):

                        pyautogui.press(
                            "volumedown"
                        )

                        show_text(
                            frame,
                            "VOLUME DOWN",
                            40
                        )

                        last_volume_time = (
                            time.time()
                        )


                # =============================================
                # OPEN PALM = PLAY / PAUSE
                # =============================================

                if fingers == 4:

                    if (
                        time.time()
                        - last_media_time
                        > GESTURE_COOLDOWN
                    ):

                        pyautogui.press(
                            "playpause"
                        )

                        show_text(
                            frame,
                            "PLAY / PAUSE",
                            40
                        )

                        last_media_time = (
                            time.time()
                        )


                # =============================================
                # FOUR FINGERS + THUMB CLOSE = SCREENSHOT
                # =============================================

                thumb_index_distance = distance(
                    thumb,
                    index
                )


                if (
                    fingers == 4
                    and
                    thumb_index_distance
                    < 0.07
                ):

                    if (
                        time.time()
                        - last_screenshot_time
                        > GESTURE_COOLDOWN
                    ):

                        screenshot = (
                            pyautogui.screenshot()
                        )


                        filename = (
                            "screenshot_"
                            + str(
                                int(
                                    time.time()
                                )
                            )
                            + ".png"
                        )


                        screenshot.save(
                            filename
                        )


                        print(
                            "Screenshot saved:",
                            filename
                        )


                        show_text(
                            frame,
                            "SCREENSHOT TAKEN",
                            40
                        )


                        last_screenshot_time = (
                            time.time()
                        )


                # =============================================
                # FIST HOLD = LOCK COMPUTER
                # =============================================

                if fingers == 0:

                    if fist_start_time is None:

                        fist_start_time = (
                            time.time()
                        )


                    elif (
                        time.time()
                        - fist_start_time
                        > 2
                    ):

                        if (
                            time.time()
                            - last_lock_time
                            > 5
                        ):

                            print(
                                "Locking computer..."
                            )


                            os.system(
                                "rundll32.exe "
                                "user32.dll,"
                                "LockWorkStation"
                            )


                            last_lock_time = (
                                time.time()
                            )


                            fist_start_time = None


                else:

                    fist_start_time = None


            # =================================================
            # PAUSED
            # =================================================

            else:

                show_text(
                    frame,
                    "MOUSE PAUSED",
                    40
                )


            # =================================================
            # DRAW HAND LANDMARKS
            # =================================================

            for point in hand:

                x = int(
                    point.x
                    * CAMERA_WIDTH
                )

                y = int(
                    point.y
                    * CAMERA_HEIGHT
                )


                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 0),
                    -1
                )


        # =================================================
        # INFORMATION
        # =================================================

        show_text(
            frame,
            f"Cursor Speed: {cursor_speed:.1f}",
            420
        )

        show_text(
            frame,
            "[ = Slower   ] = Faster",
            450
        )

        show_text(
            frame,
            "P = Pause   Q = Exit",
            480
        )


        # =================================================
        # KEYBOARD CONTROLS
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        # Quit
        if key == ord("q"):

            running = False


        # Cursor slower
        elif key == ord("["):

            cursor_speed = max(
                0.2,
                cursor_speed - 0.1
            )

            print(
                "Cursor speed:",
                cursor_speed
            )


        # Cursor faster
        elif key == ord("]"):

            cursor_speed = min(
                3.0,
                cursor_speed + 0.1
            )

            print(
                "Cursor speed:",
                cursor_speed
            )


        # Pause / resume
        elif key == ord("p"):

            mouse_paused = (
                not mouse_paused
            )

            print(
                "Mouse paused:",
                mouse_paused
            )


        # Show window
        cv2.imshow(
            "AI Virtual Mouse",
            frame
        )


# =====================================================
# CLEANUP
# =====================================================

if mouse_holding:

    pyautogui.mouseUp()


cap.release()

cv2.destroyAllWindows()

print("Virtual Mouse stopped.")
