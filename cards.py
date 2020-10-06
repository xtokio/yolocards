import cv2
import numpy as np

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

modelConfiguration = "yolov3/yolocards.cfg"
modelWeights = "yolov3/yolocards_608.weights"
classesFile = "yolov3/cards.names"
classNames = []

# Read file
with open(classesFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confidences = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confidences,confThreshold,nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),1)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confidences[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,255),1)

def showVideo(cap):
    while True:
        success, img = cap.read()
        if success:
            blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
            net.setInput(blob)
            layerNames = net.getLayerNames()
            outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

            outputs = net.forward(outputNames)
            findObjects(outputs,img)

            cv2.imshow("Video",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

def showImage(img):
    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    cv2.imshow("Image",img)
    cv2.waitKey(0)

# Read Camera
# cap = cv2.VideoCapture(0)
# Width
# cap.set(3,640)
# Heigth
# cap.set(4,480)
# Brightness
# cap.set(10,100)
# showVideo(cap)

# Read Video file
# cap = cv2.VideoCapture("video/2c.avi")
# showVideo(cap)

# Read Image file
img = cv2.imread("img/cards_02.jpg")
showImage(img)