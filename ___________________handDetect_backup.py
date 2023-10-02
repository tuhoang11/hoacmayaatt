import cv2
import mediapipe
import numpy as np
from detection import predict_rgb_image_vgg

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)



with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=1) as hands:
    count = 0
    while (True):
      hand = []
      ret, frame = capture.read()
      mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
      results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      landMarkList = []
      if results.multi_hand_landmarks != None:
          for handLandmarks in results.multi_hand_landmarks:
              x_max = 0
              y_max = 0
              x_min = frameWidth
              y_min = frameHeight
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
                  vien =  int(y_max / 25)
                  x1 = x_min-vien
                  y1 = y_min-vien
                  x2 = x_max+vien
                  y2 = y_max+vien
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                  hand = frame[y1:y2, x1:x2]
                  if pixelCoordinatesLandmark != None:
                      drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
                      handx = pixelCoordinatesLandmark[0]
                      handy = pixelCoordinatesLandmark[1]
                      cv2.circle(mask, pixelCoordinatesLandmark, int(vien/2), (255, 255, 255), -1)
                      for xx in range(0,4):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx+1][1], landMarkList[xx+1][2]), (255, 255, 255), vien)
                      for xx in range(5,8):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx+1][1], landMarkList[xx+1][2]), (255, 255, 255), vien)
                      for xx in range(9,12):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx+1][1], landMarkList[xx+1][2]), (255, 255, 255), vien)
                      for xx in range(13,16):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx+1][1], landMarkList[xx+1][2]), (255, 255, 255), vien)
                      for xx in range(17,20):
                          cv2.line(mask, (landMarkList[xx][1], landMarkList[xx][2]),
                                   (landMarkList[xx+1][1], landMarkList[xx+1][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[5][1], landMarkList[5][2]),
                               (landMarkList[9][1], landMarkList[9][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[9][1], landMarkList[9][2]),
                               (landMarkList[13][1], landMarkList[13][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[13][1], landMarkList[13][2]),
                               (landMarkList[17][1], landMarkList[17][2]), (255, 255, 255), vien)
                      cv2.line(mask, (landMarkList[0][1], landMarkList[0][2]),
                               (landMarkList[17][1], landMarkList[17][2]), (255, 255, 255), vien)
              if pixelCoordinatesLandmark != None:
                  mask = mask[y1:y2, x1:x2]
                  cv2.imshow('mask', mask)
                  thresh = mask
                  if (thresh is not None):
                      # Dua vao mang de predict
                      target = np.stack((thresh,) * 3, axis=-1)
                      target = cv2.resize(target, (224, 224))
                      target = target.reshape(1, 224, 224, 3)
                      prediction, score = predict_rgb_image_vgg(target)

                      # Neu probality > nguong du doan thi hien thi
                      print(score, prediction)
                      cv2.putText(frame, "Sign:" + prediction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                  (0, 0, 255), 10, lineType=cv2.LINE_AA)


      cv2.imshow('Test hand', frame)
      if cv2.waitKey(1) == 27:
          break

cv2.destroyAllWindows()
capture.release()