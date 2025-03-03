'''
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow("MediaPipe Pose", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
'''
import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
#pose = mp_pose.Pose()
pose = mp_pose.Pose(model_complexity=0)  # 0 for Lite, 1 for Full, 2 for Heavy

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Start time for the entire loop
    loop_start_time = time.time()

    # Capture frame
    capture_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    capture_time = time.time() - capture_start_time

    # Convert the frame to RGB
    rgb_start_time = time.time()
    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (256, 256))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_time = time.time() - rgb_start_time

    # Process the frame with MediaPipe Pose
    inference_start_time = time.time()
    results = pose.process(rgb_frame)
    inference_time = time.time() - inference_start_time

    # Draw pose landmarks on the frame
    draw_start_time = time.time()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    draw_time = time.time() - draw_start_time

    # Display the frame
    display_start_time = time.time()
    cv2.imshow("MediaPipe Pose", frame)
    display_time = time.time() - display_start_time

    # Calculate total loop time
    loop_time = time.time() - loop_start_time

    # Print timings
    print(f"Capture: {capture_time:.3f}s, RGB Conversion: {rgb_time:.3f}s, "
          f"Inference: {inference_time:.3f}s, Draw: {draw_time:.3f}s, "
          f"Display: {display_time:.3f}s, Total Loop: {loop_time:.3f}s")

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
