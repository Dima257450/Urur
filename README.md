```
import cv2
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
  frameOpencvDnn = frame.copy()
  frameHeight = frameOpencvDnn.shape[0]
  frameWidth = frameOpencvDnn.shape[1]
  blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
  net.setInput(blob)
  detections = net.forward()
  faceBoxes = []
  for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        faceBoxes.append([x1, y1, x2, y2])
        cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

  return frameOpencvDnn, faceBoxes

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male ','Female']
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderNet=cv2.dnn.readNet(genderModel,genderProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)

imagePath = "dataset/pexels-jack-sparrow-4046771.jpg" 
frame = cv2.imread(imagePath)

down_width = 640
down_height = 400
down_points = (down_width, down_height)
frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)

if frame is None:
  print("Изображение не загружено")
  exit()

resultImg, faceBoxes = highlightFace(faceNet, frame)
for faceBox in faceBoxes:
  face=frame[max(0,faceBox[1]):
    min(faceBox[3],frame.shape[0]-1),max(0,faceBox[0])
    :min(faceBox[2], frame.shape[1]-1)]
  blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
  genderNet.setInput(blob)
  genderPreds=genderNet.forward()
  gender=genderList[genderPreds[0].argmax()]
  print(f'Пол: {gender}')

  ageNet.setInput(blob)
  agePreds=ageNet.forward()
  age=ageList[agePreds[0].argmax()]
  print(f'Лет: {age[1:-1]} годов')

  cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

if not faceBoxes:
  print("Лица не распознаны")
else:
  #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #cv2.imshow('', gray_image)
  #blur = cv2.GaussianBlur(resultImg, (21, 11), 0)
  '''
  def trackbar(x):    
    ret, img1 = cv2.threshold(resultImg, x, 255, cv2.THRESH_BINARY)
    ret, img2 = cv2.threshold(resultImg, x, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('window', np.hstack([resultImg, img1, img2]))

    text = f'threshold={x}, mode=BINARY, BINARY_INV'
    cv2.displayOverlay('window', text, 1000)

  cv2.imshow('', resultImg)
  trackbar(100)
  '''

  #img1 = cv2.Canny(resultImg, 100, 200)
  #cv2.imshow('', img1)
  cv2.imshow('', resultImg)
  cv2.waitKey(0)

cv2.destroyAllWindows()
