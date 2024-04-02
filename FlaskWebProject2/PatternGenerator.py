import cv2
def GenerateHaarCasecadePatternFile(filename):
    camera =cv2.VideoCapture(filename)
    while True:
        status ,frame =  camera.read()
        if(status == False):
            print(frame)

