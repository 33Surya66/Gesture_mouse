import cv2
import mediapipe
import pyautogui
import numpy as np
import time
import math

# Initialize MediaPipe Hand tracking
mp_hands = mediapipe.solutions.hands
capture_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
drawing_option = mediapipe.solutions.drawing_utils
drawing_styles = mediapipe.solutions.drawing_styles

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize camera
camera = cv2.VideoCapture(0)

# Smoothing factor for mouse movement
smoothing = 5
prev_mouse_x, prev_mouse_y = 0, 0

# Mode settings
MODE = "MOUSE"  # "MOUSE", "SCROLL", "DRAW", "VOLUME"
CLICK_COOLDOWN = 0.5  # seconds between clicks
last_click_time = 0
draw_points = []
is_drawing = False

# Settings
SHOW_FPS = True
SMOOTHING_ENABLED = True
GESTURE_INSTRUCTIONS = True

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_hand_gesture(landmarks):
    # Get fingertip landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Get finger base landmarks
    index_base = landmarks[5]
    middle_base = landmarks[9]
    ring_base = landmarks[13]
    pinky_base = landmarks[17]
    
    # Check if fingers are extended
    fingers_extended = []
    
    # Thumb is extended if it's to the side of the thumb base
    thumb_extended = thumb_tip.x < landmarks[3].x if landmarks[0].x < landmarks[9].x else thumb_tip.x > landmarks[3].x
    fingers_extended.append(thumb_extended)
    
    # Other fingers are extended if their tips are above their bases
    fingers_extended.append(index_tip.y < index_base.y)
    fingers_extended.append(middle_tip.y < middle_base.y)
    fingers_extended.append(ring_tip.y < ring_base.y)
    fingers_extended.append(pinky_tip.y < pinky_base.y)
    
    # Return gesture based on extended fingers
    if all(fingers_extended):
        return "OPEN_HAND"
    elif fingers_extended[1] and not any(fingers_extended[2:]):
        return "POINTER"
    elif fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]):
        return "VICTORY"
    elif fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:]):
        return "PINCH"
    elif not any(fingers_extended):
        return "FIST"
    elif not fingers_extended[0] and all(fingers_extended[1:]):
        return "FOUR_FINGERS"
    elif fingers_extended[0] and not any(fingers_extended[1:]):
        return "THUMB_ONLY"
    else:
        return "OTHER"

# FPS calculation
prev_time = 0
curr_time = 0

while True:
    success, image = camera.read()
    if not success:
        print("Failed to capture image")
        break
        
    # Flip the image horizontally
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hand tracking
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks
    
    # Show current mode
    cv2.putText(image, f"Mode: {MODE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    # Display FPS
    if SHOW_FPS:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
    
    # Draw instruction based on mode
    if GESTURE_INSTRUCTIONS:
        instructions = {
            "MOUSE": "Index finger: Move cursor | Thumb-Index pinch: Click",
            "SCROLL": "Index up/down: Scroll | Fist: Back to mouse mode",
            "DRAW": "Index finger: Draw | Fist: Clear | Open hand: Back to mouse",
            "VOLUME": "Index-Thumb distance: Volume | Fist: Back to mouse"
        }
        cv2.putText(image, instructions[MODE], (10, image_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw in drawing mode
    if MODE == "DRAW" and draw_points:
        for i in range(1, len(draw_points)):
            cv2.line(image, draw_points[i-1], draw_points[i], (0, 0, 255), 5)
    
    if all_hands:
        for hand in all_hands:
            # Draw hand landmarks
            drawing_option.draw_landmarks(
                image, 
                hand, 
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
            
            landmarks = hand.landmark
            
            # Extract key points
            thumb_tip = (int(landmarks[4].x * image_width), int(landmarks[4].y * image_height))
            index_tip = (int(landmarks[8].x * image_width), int(landmarks[8].y * image_height))
            middle_tip = (int(landmarks[12].x * image_width), int(landmarks[12].y * image_height))
            wrist = (int(landmarks[0].x * image_width), int(landmarks[0].y * image_height))
            
            # Get hand gesture
            gesture = get_hand_gesture(landmarks)
            cv2.putText(image, f"Gesture: {gesture}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mode switching
            if gesture == "FOUR_FINGERS" and MODE != "MOUSE":
                MODE = "MOUSE"
            elif gesture == "VICTORY" and MODE != "SCROLL":
                MODE = "SCROLL"
            elif gesture == "OPEN_HAND" and MODE != "DRAW":
                MODE = "DRAW"
                draw_points = []
            elif gesture == "THUMB_ONLY" and MODE != "VOLUME":
                MODE = "VOLUME"
            
            # Mouse mode
            if MODE == "MOUSE":
                # Use index finger for mouse movement
                if landmarks[8].visibility > 0.5:
                    mouse_x = int(screen_width * landmarks[8].x)
                    mouse_y = int(screen_height * landmarks[8].y)
                    
                    # Apply smoothing
                    if SMOOTHING_ENABLED:
                        mouse_x = prev_mouse_x + (mouse_x - prev_mouse_x) // smoothing
                        mouse_y = prev_mouse_y + (mouse_y - prev_mouse_y) // smoothing
                    
                    pyautogui.moveTo(mouse_x, mouse_y)
                    prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                    
                    # Highlight the control point
                    cv2.circle(image, index_tip, 10, (0, 0, 255), -1)
                
                # Click with thumb and index finger pinch
                distance = calculate_distance(thumb_tip[0], thumb_tip[1], index_tip[0], index_tip[1])
                if distance < 30:
                    current_time = time.time()
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = current_time
                        cv2.putText(image, "Click!", (index_tip[0], index_tip[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Right click with thumb and middle finger pinch
                distance2 = calculate_distance(thumb_tip[0], thumb_tip[1], middle_tip[0], middle_tip[1])
                if distance2 < 30:
                    current_time = time.time()
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_click_time = current_time
                        cv2.putText(image, "Right Click!", (middle_tip[0], middle_tip[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Scroll mode
            elif MODE == "SCROLL":
                # Use index finger for scrolling
                if gesture == "POINTER":
                    # Vertical scrolling
                    if landmarks[8].y < 0.3:  # Upper part of the screen
                        pyautogui.scroll(10)  # Scroll up
                        cv2.putText(image, "Scroll Up", (index_tip[0], index_tip[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif landmarks[8].y > 0.7:  # Lower part of the screen
                        pyautogui.scroll(-10)  # Scroll down
                        cv2.putText(image, "Scroll Down", (index_tip[0], index_tip[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Exit scroll mode with fist
                if gesture == "FIST":
                    MODE = "MOUSE"
            
            # Draw mode
            elif MODE == "DRAW":
                if gesture == "POINTER":
                    if not is_drawing:
                        is_drawing = True
                        draw_points = [index_tip]
                    else:
                        draw_points.append(index_tip)
                else:
                    is_drawing = False
                
                # Clear drawing with fist
                if gesture == "FIST":
                    draw_points = []
                    cv2.putText(image, "Drawing Cleared", (wrist[0], wrist[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Volume control mode
            elif MODE == "VOLUME":
                # Use thumb and index distance for volume control
                distance = calculate_distance(thumb_tip[0], thumb_tip[1], index_tip[0], index_tip[1])
                volume_level = int(np.interp(distance, [30, 200], [0, 100]))
                
                # Draw volume bar
                cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(image, (50, 400 - int(volume_level * 2.5)), (85, 400), (255, 0, 0), -1)
                cv2.putText(image, f"{volume_level}%", (90, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Set system volume (requires additional library on some systems)
                try:
                    volume_level = max(0, min(100, volume_level))
                    pyautogui.press('volumemute')  # Reset volume
                    if volume_level > 0:
                        for _ in range(int(volume_level / 2)):
                            pyautogui.press('volumeup')
                except:
                    pass
                
                # Exit volume mode with fist
                if gesture == "FIST":
                    MODE = "MOUSE"
    
    # Display image
    cv2.imshow("Hand Gesture Controls", image)
    
    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('m'):
        MODE = "MOUSE"
    elif key == ord('s'):
        MODE = "SCROLL"
    elif key == ord('d'):
        MODE = "DRAW"
        draw_points = []
    elif key == ord('v'):
        MODE = "VOLUME"
    elif key == ord('f'):
        SHOW_FPS = not SHOW_FPS
    elif key == ord('g'):
        SMOOTHING_ENABLED = not SMOOTHING_ENABLED
    elif key == ord('h'):
        GESTURE_INSTRUCTIONS = not GESTURE_INSTRUCTIONS

# Release resources
camera.release()
cv2.destroyAllWindows()