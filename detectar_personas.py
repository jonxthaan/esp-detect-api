import cv2

def contar_personas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 100:
        return 1
    else:
        return 0
