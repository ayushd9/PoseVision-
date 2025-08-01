import cv2
import cv2 as cv
import mediapipe as mp
import time
import cvzone
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

def action(l):
    pass

mpDraw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # Capture video from the laptop's webcam

pTime = 0
padd = 25
action1 = ["walk", "run", "surrender"]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    landmarks = []
    try:
        landmark1 = results.pose_landmarks.landmark
    except:
        pass

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)

            landmarks.append(((int(cx)), (int(cy)), (lm.z * w)))
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        x_cor = np.array(landmarks)[:, 0]
        y_cor = np.array(landmarks)[:, 1]

        x1 = int(np.min(x_cor) - padd)
        y1 = int(np.min(y_cor) - padd)
        x2 = int(np.max(x_cor) + padd)
        y2 = int(np.max(x_cor) + padd - padd)
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.addWeighted(img.copy(), 0.5, img, 0.5, 0, img)

    # get coordinates
    lshoulder = [landmark1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmark1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    lelbow = [landmark1[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
              landmark1[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    lwrist = [landmark1[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
              landmark1[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    rshoulder = [landmark1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmark1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    relbow = [landmark1[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
              landmark1[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    rwrist = [landmark1[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
              landmark1[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    lhip = [landmark1[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmark1[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    lknee = [landmark1[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
             landmark1[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    lankle = [landmark1[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
              landmark1[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    rhip = [landmark1[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmark1[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    rknee = [landmark1[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
             landmark1[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    rankle = [landmark1[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
              landmark1[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # angle calculate
    anglel1 = 360 - calculate_angle(lshoulder, lelbow, lwrist)
    angler1 = 360 - calculate_angle(rshoulder, relbow, rwrist)
    anglel2 = calculate_angle(lelbow, lshoulder, lhip)
    angler2 = calculate_angle(relbow, rshoulder, rhip)
    anglel3 = 360 - calculate_angle(lshoulder, lhip, lknee)
    angler3 = 360 - calculate_angle(rshoulder, rhip, rknee)
    anglel4 = 360 - calculate_angle(lhip, lknee, lankle)
    angler4 = 360 - calculate_angle(rhip, rknee, rankle)
    act_angle = [anglel1, angler1, anglel2, angler2, anglel3, angler3, anglel4, angler4]

    action(act_angle)

    # visualize
    cv2.putText(img, str(anglel1), tuple(np.multiply(lelbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 18, 18), 2)
    cv2.putText(img, str(angler1), tuple(np.multiply(relbow, [640, 580]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 18, 201), 2)
    cv2.putText(img, str(anglel2), tuple(np.multiply(lshoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (90, 18, 255), 2)
    cv2.putText(img, str(angler2), tuple(np.multiply(rshoulder, [640, 580]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (18, 240, 255), 2)

    cv2.putText(img, str(anglel3), tuple(np.multiply(lhip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (18, 255, 74), 2)
    cv2.putText(img, str(angler3), tuple(np.multiply(rhip, [640, 580]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (216, 255, 18), 2)
    cv2.putText(img, str(anglel4), tuple(np.multiply(lknee, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 18, 18), 2)
    cv2.putText(img, str(angler4), tuple(np.multiply(rknee, [640, 580]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 18, 201), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
