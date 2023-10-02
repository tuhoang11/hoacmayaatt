import cv2
import mediapipe
import numpy as np
from detection import predict_rgb_image_vgg

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

mp_pose = mediapipe.solutions.pose

capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

class PoseLandmark():
  """The 33 pose landmarks."""
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=1) as hands:
    pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    count = 0
    while (True):
      hand = []
      ret, frame = capture.read()
      mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
      results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      pose_result = pose.process(frame)
      POSE_CONNECTIONS = frozenset([
          (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
          (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
          (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER),
          (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
          (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
          (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE),
          (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER),
          (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR),
          (PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT),
          (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),
          (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
          (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
          (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
          (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
          (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
          (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
          (PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_HIP),
          (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
          (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
          (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
          (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
          (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
          (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
          (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX),
          (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX),
          (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
          (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
      ])
      # drawingModule.draw_landmarks(frame, pose_result.pose_landmarks, POSE_CONNECTIONS)
      #Lấy landmark body
      poseLandMarkList = []
      x_max = 0
      y_max = 0
      x_min = frameWidth
      y_min = frameHeight
      if pose_result.pose_landmarks != None:
          keypoints = []
          for data_point in pose_result.pose_landmarks.landmark:
              x, y = int(data_point.x * frameWidth), int(data_point.y * frameHeight)
              keypoints.append([x, y])
      landMarkList = []
      if results.multi_hand_landmarks != None:
          for handLandmarks in results.multi_hand_landmarks:
              for point in handsModule.HandLandmark:
                  for id, landMark in enumerate(handLandmarks.landmark):
                      # landMark holds x,y,z ratios of single landmark
                      imgH, imgW, imgC = frame.shape  # height, width, channel for image
                      xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                      landMarkList.append([id, xPos, yPos])
                  for lm in handLandmarks.landmark:
                      x, y = int(lm.x * frameWidth), int(lm.y * frameHeight)
                      if x > x_max:
                          x_max = x
                      if x < x_min:
                          x_min = x
                      if y > y_max:
                          y_max = y
                      if y < y_min:
                          y_min = y
                  normalizedLandmark = handLandmarks.landmark[point]
                  pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                            normalizedLandmark.y,
                                                                                            frameWidth, frameHeight)
                  # cv2.circle(frame, pixelCoordinatesLandmark, 5, (0, 255, 0), -1)
                  vien =  int(y_max / 120)
                  if vien < 1:
                      vien = 1
                  x1 = x_min-vien
                  y1 = y_min-vien
                  x2 = x_max+vien
                  y2 = y_max+vien
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                  hand = frame[y1:y2, x1:x2]
                  if pixelCoordinatesLandmark != None and pose_result.pose_landmarks != None:
                      drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
                      handx = pixelCoordinatesLandmark[0]
                      handy = pixelCoordinatesLandmark[1]
                      cv2.circle(mask, pixelCoordinatesLandmark, int(vien * 1.5), (255, 255, 255), -1)
                      for xx in range(0, 4):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx + 1][1], landMarkList[xx + 1][2]), (255, 255, 255), vien)
                      for xx in range(5, 8):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx + 1][1], landMarkList[xx + 1][2]), (255, 255, 255), vien)
                      for xx in range(9, 12):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx + 1][1], landMarkList[xx + 1][2]), (255, 255, 255), vien)
                      for xx in range(13, 16):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx + 1][1], landMarkList[xx + 1][2]), (255, 255, 255), vien)
                      for xx in range(17, 20):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx + 1][1], landMarkList[xx + 1][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[5][1], landMarkList[5][2]),
                               (landMarkList[9][1], landMarkList[9][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[9][1], landMarkList[9][2]),
                               (landMarkList[13][1], landMarkList[13][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[13][1], landMarkList[13][2]),
                               (landMarkList[17][1], landMarkList[17][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[0][1], landMarkList[0][2]),
                               (landMarkList[17][1], landMarkList[17][2]), (255, 255, 255), vien)

              if pixelCoordinatesLandmark != None and mask.all() != None:
                  mask = mask[y1:y2, x1:x2]
                  thresh = mask
                  if thresh is not None:
                      try:
                          cv2.imshow("thresh", thresh)
                          # Dua vao mang de predict
                          target = np.stack((thresh,) * 3, axis=-1)
                          target = cv2.resize(target, (224, 224))
                          target = target.reshape(1, 224, 224, 3)
                          prediction, score = predict_rgb_image_vgg(target)
                          # Neu probality > nguong du doan thi hien thi
                          if score >= 99.9:
                            cv2.putText(frame, prediction, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                      (0, 0, 255), 4, lineType=cv2.LINE_AA)
                      except:
                          pass


      cv2.imshow('Frame', frame)
      if cv2.waitKey(1) == 27:
          break

cv2.destroyAllWindows()
capture.release()