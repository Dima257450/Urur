```
import cv2

# функция определения лиц
def highlightFace(net, frame, conf_threshold=0.7):
    # делаем копию текущего кадра
    frameOpencvDnn = frame.copy()
    # высота и ширина кадра
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # преобразуем картинку в двоичный пиксельный объект
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц
    detections = net.forward()
    # переменная для рамок вокруг лица
    faceBoxes = []
    # перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # получаем результат вычислений для очередного элемента
        confidence = detections[0, 0, i, 2]
        # если результат превышает порог срабатывания — это лицо
        if confidence > conf_threshold:
            # формируем координаты рамки
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            # добавляем их в общую переменную
            faceBoxes.append([x1, y1, x2, y2])
            # рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    # возвращаем кадр с рамками
    return frameOpencvDnn, faceBoxes

# загружаем веса для распознавания лиц
faceProto = "opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel = "opencv_face_detector_uint8.pb"

# запускаем нейросеть по распознаванию лиц
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# загружаем изображение с диска
imagePath = "path_to_your_image.jpg"  # Укажите путь к вашему изображению
frame = cv2.imread(imagePath)

# проверяем, успешно ли загружено изображение
if frame is None:
    print("Ошибка: изображение не загружено")
    exit()

# распознаём лица в изображении
resultImg, faceBoxes = highlightFace(faceNet, frame)
# если лиц нет
if not faceBoxes:
    # выводим в консоли, что лицо не найдено
    print("Лица не распознаны")
else:
    # выводим изображение с рамками
    cv2.imshow("Face detection", resultImg)
    # ждём нажатия клавиши для закрытия окна
    cv2.waitKey(0)

# закрываем все окна
cv2.destroyAllWindows()
