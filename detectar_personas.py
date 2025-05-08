import cv2
import numpy as np

# Leer nombres de clases
classNames = []
with open("coco.names", "rt") as f:
    classNames = f.read().rstrip('\n').split('\n')

# Modelo preentrenado de detección
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

# Configurar red neuronal
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# ID de clase "person"
person_class_id = classNames.index("persona") + 1

def contar_personas(img):
    person_count = 0
    classIds, confs, bbox = net.detect(img, confThreshold=0.6)

    if len(classIds) != 0:
        for classId in classIds.flatten():
            if classId == person_class_id:
                person_count += 1

    return person_count

