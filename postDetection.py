import cv2
import mediapipe as mp
import time
import ctypes
import numpy as np
import threading
import winsound 

mpDraw = mp.solutions.drawing_utils
mPose = mp.solutions.pose
pose = mPose.Pose()

cap = cv2.VideoCapture(0)

pTime = 0
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

reference_landmarks = None

def get_landmarks(landmarks):
    return np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark])

def normalize_landmarks(landmarks):
    torso_center = (landmarks[11] + landmarks[12]) / 2
    normalized = landmarks - torso_center  
    
    max_distance = np.linalg.norm(landmarks[11] - landmarks[12])  
    return normalized / max_distance  

def compare_poses(current_landmarks, reference_landmarks):
    if current_landmarks is None or reference_landmarks is None:
        return None
    
    current_landmarks = normalize_landmarks(current_landmarks)
    reference_landmarks = normalize_landmarks(reference_landmarks)
    
    distances = np.linalg.norm(current_landmarks - reference_landmarks, axis=1)
    
    threshold = 0.2  
    score = np.mean(distances < threshold) * 100  
    return score

def play_beep():
    winsound.Beep(1000, 500)  

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

        if reference_landmarks is None:
            reference_landmarks = current_landmarks
            print("Reference pose captured.")

        score = compare_poses(current_landmarks, reference_landmarks)
        if score is not None:
            if score > 95:
                feedback = "Accurate"
                color = (0, 255, 0) 
            elif 85 < score <= 95:
                feedback = "Good"
                color = (255, 255, 0) 
            elif 50 <= score <= 70:
                feedback = "Need More Improvement"
                color = (0, 165, 255) 
            else:
                feedback = "Incorrect! Correct Your Pose"
                color = (0, 0, 255) 
                
                threading.Thread(target=play_beep).start()

            cv2.putText(img, feedback, (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
            cv2.putText(img, f'Pose Accuracy: {int(score)}%', (50, 150), 
                        cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    img = cv2.resize(img, (screen_width, screen_height))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
