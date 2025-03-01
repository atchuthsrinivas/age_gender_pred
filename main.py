import cv2

def faceBox(faceNet, frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    boxes=[]
    for i in range(detection.shape[2]):
        confidence=detection[0, 0, i, 2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            boxes.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, boxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel, ageProto)
genderNet=cv2.dnn.readNet(genderModel, genderProto)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


video=cv2.VideoCapture(0)

padding=20

while True:
    ret, frame=video.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    frame, boxes = faceBox(faceNet,frame)
    
    for box in boxes:
        # face=frame[box[1]:box[3], box[0]:box[2]]
        face = frame[max(0, box[1]-padding):min(box[3]+padding,frame.shape[0]-1), max(0, box[0]-padding):min(box[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]
        
        
        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]
        
        
        label="{},{}".format(gender, age)
        cv2.rectangle(frame, (box[0], box[1]-30), (box[2], box[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225,225,255), 2,cv2.LINE_AA)
    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1) & 0xFF
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()