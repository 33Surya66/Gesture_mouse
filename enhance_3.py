import cv2
import mediapipe
import pyautogui
import numpy as np
import time
import math

# Disable pyautogui's failsafe to allow cursor movement across entire screen
pyautogui.FAILSAFE = False

mp_hands = mediapipe.solutions.hands
capture_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
drawing_option = mediapipe.solutions.drawing_utils
drawing_styles = mediapipe.solutions.drawing_styles

# Get actual laptop screen dimensions
screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)

# Get camera dimensions
_, test_frame = camera.read()
if test_frame is not None:
    camera_height, camera_width, _ = test_frame.shape
else:
    # Fallback camera dimensions if we can't read a frame
    camera_width, camera_height = 640, 480

# Improved smoothing for better cursor control
smoothing = 7
prev_mouse_x, prev_mouse_y = pyautogui.position()  # Initialize with current cursor position

MODE = "MOUSE"
CLICK_COOLDOWN = 0.5
last_click_time = 0
draw_points = []
is_drawing = False

SHOW_FPS = True
SMOOTHING_ENABLED = True
GESTURE_INSTRUCTIONS = True

# Add a small buffer around screen edges to make corner access easier
EDGE_BUFFER = 20

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_hand_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    index_base = landmarks[5]
    middle_base = landmarks[9]
    ring_base = landmarks[13]
    pinky_base = landmarks[17]
    
    fingers_extended = []
    
    # Improved thumb extension detection based on hand orientation
    wrist = landmarks[0]
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    
    # Calculate hand direction vector
    hand_direction_x = thumb_mcp.x - wrist.x
    
    # Determine if hand is left or right based on thumb position relative to wrist
    is_right_hand = hand_direction_x < 0
    
    # Updated thumb extended logic based on hand orientation
    if is_right_hand:
        thumb_extended = thumb_tip.x < thumb_ip.x
    else:
        thumb_extended = thumb_tip.x > thumb_ip.x
    
    fingers_extended.append(thumb_extended)
    
    # More accurate finger extension detection with additional checks
    fingers_extended.append(index_tip.y < index_base.y and calculate_distance(index_tip.x, index_tip.y, index_base.x, index_base.y) > 0.05)
    fingers_extended.append(middle_tip.y < middle_base.y and calculate_distance(middle_tip.x, middle_tip.y, middle_base.x, middle_base.y) > 0.05)
    fingers_extended.append(ring_tip.y < ring_base.y and calculate_distance(ring_tip.x, ring_tip.y, ring_base.x, ring_base.y) > 0.05)
    fingers_extended.append(pinky_tip.y < pinky_base.y and calculate_distance(pinky_tip.x, pinky_tip.y, pinky_base.x, pinky_base.y) > 0.05)
    
    # Improved gesture recognition logic
    if all(fingers_extended):
        return "OPEN_HAND"
    elif fingers_extended[1] and not any(fingers_extended[2:]) and not fingers_extended[0]:
        return "POINTER"
    elif fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]) and not fingers_extended[0]:
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

# Function to map camera coordinates to full screen coordinates
def map_to_screen(x_cam, y_cam):
    # Map from camera space (0-1) to screen space with padding
    x_screen = np.interp(x_cam, [0.1, 0.9], [-EDGE_BUFFER, screen_width + EDGE_BUFFER])
    y_screen = np.interp(y_cam, [0.1, 0.9], [-EDGE_BUFFER, screen_height + EDGE_BUFFER])
    
    # Clamp values to valid screen coordinates
    x_screen = max(0, min(screen_width - 1, x_screen))
    y_screen = max(0, min(screen_height - 1, y_screen))
    
    return int(x_screen), int(y_screen)

prev_time = 0
curr_time = 0

while True:
    success, image = camera.read()
    if not success:
        print("Failed to capture image")
        break
        
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks
    
    cv2.putText(image, f"Mode: {MODE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    if SHOW_FPS:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
    
    if GESTURE_INSTRUCTIONS:
        instructions = {
            "MOUSE": "Index finger: Move cursor | Thumb-Index pinch: Click | Thumb-Middle pinch: Right Click",
            "SCROLL": "Index up/down: Scroll | Fist: Back to mouse mode",
            "DRAW": "Index finger: Draw | Fist: Clear | Open hand: Back to mouse"
        }
        cv2.putText(image, instructions[MODE], (10, image_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if MODE == "DRAW" and draw_points:
        for i in range(1, len(draw_points)):
            cv2.line(image, draw_points[i-1], draw_points[i], (0, 0, 255), 5)
    
    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(
                image, 
                hand, 
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
            
            landmarks = hand.landmark
            
            thumb_tip = (int(landmarks[4].x * image_width), int(landmarks[4].y * image_height))
            index_tip = (int(landmarks[8].x * image_width), int(landmarks[8].y * image_height))
            middle_tip = (int(landmarks[12].x * image_width), int(landmarks[12].y * image_height))
            wrist = (int(landmarks[0].x * image_width), int(landmarks[0].y * image_height))
            
            gesture = get_hand_gesture(landmarks)
            cv2.putText(image, f"Gesture: {gesture}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if gesture == "FOUR_FINGERS" and MODE != "MOUSE":
                MODE = "MOUSE"
            elif gesture == "VICTORY" and MODE != "SCROLL":
                MODE = "SCROLL"
            elif gesture == "OPEN_HAND" and MODE != "DRAW":
                MODE = "DRAW"
                draw_points = []
            
            if MODE == "MOUSE":
                if landmarks[8].visibility > 0.7:  # Increased visibility threshold for more stability
                    # Map camera coordinates to full screen coordinates
                    mouse_x, mouse_y = map_to_screen(landmarks[8].x, landmarks[8].y)
                    
                    if SMOOTHING_ENABLED:
                        # Enhanced smoothing algorithm for more stable cursor movement
                        mouse_x = prev_mouse_x + (mouse_x - prev_mouse_x) // smoothing
                        mouse_y = prev_mouse_y + (mouse_y - prev_mouse_y) // smoothing
                    
                    # Move mouse cursor on entire laptop screen
                    try:
                        pyautogui.moveTo(mouse_x, mouse_y)
                        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                    except Exception as e:
                        print(f"Error moving mouse: {e}")
                    
                    # Visual indicator in camera view
                    cv2.circle(image, index_tip, 10, (0, 0, 255), -1)
                    
                    # Show coordinates for debugging
                    cv2.putText(image, f"Screen: {mouse_x},{mouse_y}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Improved click detection with more precise distance threshold
                distance = calculate_distance(thumb_tip[0], thumb_tip[1], index_tip[0], index_tip[1])
                if distance < 40:  # Slightly increased for more reliable detection
                    # Visual feedback showing pinch distance
                    cv2.line(image, thumb_tip, index_tip, (0, 255, 0), 2)
                    cv2.putText(image, f"{int(distance)}", ((thumb_tip[0] + index_tip[0]) // 2, 
                                (thumb_tip[1] + index_tip[1]) // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if distance < 25:  # Narrower final threshold for actual click
                        current_time = time.time()
                        if current_time - last_click_time > CLICK_COOLDOWN:
                            try:
                                pyautogui.click()
                                last_click_time = current_time
                                cv2.putText(image, "Click!", (index_tip[0], index_tip[1] - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error clicking: {e}")
                
                # Improved right-click detection
                distance2 = calculate_distance(thumb_tip[0], thumb_tip[1], middle_tip[0], middle_tip[1])
                if distance2 < 40:  # Visual feedback threshold
                    # Visual feedback for right-click distance
                    cv2.line(image, thumb_tip, middle_tip, (255, 0, 0), 2)
                    cv2.putText(image, f"{int(distance2)}", ((thumb_tip[0] + middle_tip[0]) // 2, 
                                (thumb_tip[1] + middle_tip[1]) // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if distance2 < 25:  # Narrower threshold for actual right-click
                        current_time = time.time()
                        if current_time - last_click_time > CLICK_COOLDOWN:
                            try:
                                pyautogui.rightClick()
                                last_click_time = current_time
                                cv2.putText(image, "Right Click!", (middle_tip[0], middle_tip[1] - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error right-clicking: {e}")
            
            elif MODE == "SCROLL":
                if gesture == "POINTER":
                    # More precise scroll zones with visual feedback
                    if landmarks[8].y < 0.3:
                        scroll_speed = int(np.interp(landmarks[8].y, [0.1, 0.3], [20, 5]))
                        try:
                            pyautogui.scroll(scroll_speed)
                            cv2.putText(image, f"Scroll Up: {scroll_speed}", (index_tip[0], index_tip[1] - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error scrolling up: {e}")
                    elif landmarks[8].y > 0.7:
                        scroll_speed = int(np.interp(landmarks[8].y, [0.7, 0.9], [-5, -20]))
                        try:
                            pyautogui.scroll(scroll_speed)
                            cv2.putText(image, f"Scroll Down: {scroll_speed}", (index_tip[0], index_tip[1] - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error scrolling down: {e}")
                    
                    # Visual scroll zone indicators
                    cv2.line(image, (0, int(0.3 * image_height)), (image_width, int(0.3 * image_height)), 
                             (255, 0, 0), 2)
                    cv2.line(image, (0, int(0.7 * image_height)), (image_width, int(0.7 * image_height)), 
                             (255, 0, 0), 2)
                
                if gesture == "FIST":
                    MODE = "MOUSE"
            
            elif MODE == "DRAW":
                if gesture == "POINTER":
                    # Improved drawing with visibility check
                    if landmarks[8].visibility > 0.9:
                        # More precise index fingertip tracking for drawing
                        precise_tip = (int(landmarks[8].x * image_width), int(landmarks[8].y * image_height))
                        
                        if not is_drawing:
                            is_drawing = True
                            draw_points = [precise_tip]
                        else:
                            # Only add points if they're sufficiently different for smoother lines
                            if not draw_points or calculate_distance(draw_points[-1][0], draw_points[-1][1], 
                                                                   precise_tip[0], precise_tip[1]) > 5:
                                draw_points.append(precise_tip)
                else:
                    is_drawing = False
                
                if gesture == "FIST":
                    draw_points = []
                    cv2.putText(image, "Drawing Cleared", (wrist[0], wrist[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add status indicator showing control is active across entire screen
    cv2.rectangle(image, (image_width - 20, 10), (image_width - 10, 20), (0, 255, 0), -1)
    cv2.putText(image, "Full Screen Control Active", (image_width - 200, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show mini laptop screen representation
    margin = 10
    mini_width = 100
    mini_height = int(mini_width * (screen_height / screen_width))
    cv2.rectangle(image, (margin, image_height - mini_height - margin), 
                 (margin + mini_width, image_height - margin), (255, 255, 255), 1)
    
    # Show cursor position on mini screen
    if prev_mouse_x is not None and prev_mouse_y is not None:
        cursor_x = int(margin + (prev_mouse_x / screen_width) * mini_width)
        cursor_y = int(image_height - margin - mini_height + (prev_mouse_y / screen_height) * mini_height)
        cursor_pos = (cursor_x, cursor_y)
        cv2.circle(image, cursor_pos, 3, (0, 0, 255), -1)
    
    cv2.imshow("Hand Gesture Controls", image)
    
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
    elif key == ord('f'):
        SHOW_FPS = not SHOW_FPS
    elif key == ord('g'):
        SMOOTHING_ENABLED = not SMOOTHING_ENABLED
    elif key == ord('h'):
        GESTURE_INSTRUCTIONS = not GESTURE_INSTRUCTIONS

camera.release()
cv2.destroyAllWindows()