import cv2
import mediapipe as mp
import time
import ctypes
import numpy as np
import threading
import winsound  # Beep sound for Windows

mpDraw = mp.solutions.drawing_utils
mPose = mp.solutions.pose
pose = mPose.Pose()

# Place 0 for camera input or a file path for a video
cap = cv2.VideoCapture("5928275-uhd_3840_2160_25fps.mp4")

pTime = 0
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Initialize reference_landmarks (correct pose)
reference_landmarks = None

def get_landmarks(landmarks):
    return np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark])

def normalize_landmarks(landmarks):
    # Normalize landmarks with respect to the whole body
    # Normalize landmarks relative to the torso (landmark 11: left shoulder, 12: right shoulder)
    torso_center = (landmarks[11] + landmarks[12]) / 2
    normalized = landmarks - torso_center  # Translate pose to the origin based on torso center
    max_distance = np.linalg.norm(landmarks[11] - landmarks[12])  # Scale based on torso width
    return normalized / max_distance  # Normalize by scaling distances

def compare_poses(current_landmarks, reference_landmarks):
    if current_landmarks is None or reference_landmarks is None:
        return None
    
    # Normalize both poses before comparison
    current_landmarks = normalize_landmarks(current_landmarks)
    reference_landmarks = normalize_landmarks(reference_landmarks)
    
    # Compare using Euclidean distance between corresponding points
    distances = np.linalg.norm(current_landmarks - reference_landmarks, axis=1)
    
    # Increase tolerance for pose accuracy
    threshold = 0.2  # Adjust tolerance for comparison
    score = np.mean(distances < threshold) * 100  # Pose accuracy in percentage
    return score

def play_beep():
    winsound.Beep(1000, 500)  # Beep sound for incorrect pose

while True:
    success, img = cap.read()
    if not success:
        print("End of video or failed to read video.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mPose.POSE_CONNECTIONS)

        current_landmarks = get_landmarks(results.pose_landmarks)

        # If reference landmarks are not set, capture the first pose as reference
        if reference_landmarks is None:
            reference_landmarks = current_landmarks
            print("Reference pose captured.")

        # Compare current pose with reference pose
        score = compare_poses(current_landmarks, reference_landmarks)
        if score is not None:
            if score > 95:
                feedback = "Accurate"
                color = (0, 255, 0)  # Green
            elif 85 < score <= 95:
                feedback = "Good"
                color = (255, 255, 0)  # Yellow
            elif 50 <= score <= 70:
                feedback = "Need More Improvement"
                color = (0, 165, 255)  # Orange
            else:
                feedback = "Incorrect! Correct Your Pose"
                color = (0, 0, 255)  # Red
                
                # Play beep sound in a separate thread
                threading.Thread(target=play_beep).start()

            # Display feedback on screen
            cv2.putText(img, feedback, (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
            cv2.putText(img, f'Pose Accuracy: {int(score)}%', (50, 150), 
                        cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    # Resize the image to fit the screen
    img = cv2.resize(img, (screen_width, screen_height))

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
