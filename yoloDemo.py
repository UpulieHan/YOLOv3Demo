import cv2
import numpy as np

# create a while loop to get the frames of our webcam
# arg 0 - for the camera connected to the PC
cap = cv2.VideoCapture(0)

#width,height,Target
whT=320

confThreshold = 0.5

#the lower this is the more intensive it will be in elminating overlapping
nms_threshold = 0.3

# To run the Yolo3 model
# Yolo3 is trained on the Coco dataset
# classes = ['','',''] likewise
# But since it's a big list, to get the names of the classes(80 different classes),
classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# to check if the extraction is correct
# print(classNames)
# print(len(classNames))

# importing the model files
# 2 main components (In Tensorflow save model method both of these files are compiled together)
# 1.configurations file - parameters like learning rate,maximum batches,steps and individual convolutional layers and their params like the number of filters, strides, pads, activation functions (architecture of the network)
# 2.weights file -
# the fps is a tradeoff with the image pixel size
# YOLOv320 - 45fps - general purpose
# YOLOtiny - 220fps (trade off becuase accuracy will go down(less detectionn)) -  raspberry ppie/jetson nano
# loading the files,

#good for NVDIA GPU with all necessary SW installed
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

#Raspberry pie,Jetson Nano
# modelConfiguration = 'yolov3-tiny.cfg'
# modelWeights = 'yolov3-tiny.weights'

# create the network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# declare opencv as the backend,usage of CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(output,img):
    #height, width,channels
    hT,wT,cT = img.shape
    #3 differect list to store
    # (whenever we find a good object detection, we will put them in these lists)

    #bounding box will contain width height(x and y)
    bbox = []

    #class Ids and their confidence value
    classIds = []
    confs = []

    #looping
    for output in outputs:
        for det in output:
            #remove first 5 elements find the value of the height value
            scores = det[5:]
            #index
            classId = np.argmax(scores)
            #value of that index
            confidence = scores[classId]

            #filtering object
            if confidence > confThreshold:
                #save width,height,x,y(are in decimals so we have to multiple them by our actual image size)
                w,h=int(det[2]*wT),int(det[3]*hT)
                #the center point(divide by 2 and substract)
                x,y=int(det[0]*wT - w/2) , int(det[1]*hT - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))

    #to remove overlapping bounding boxes(non maximum surpression)
    #by finding and picking the maximum confidence value box
    #output are the indices of the bboxes to keep
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nms_threshold)
    # prinnt(indices)

    #ploting the remaining indices in a loop
    for i in indices:
        #to remove the extra bracket
        i = i[0]
        box = bbox[i]
        #extract x,y,width,height
        x,y,w,h = box[0],box[1],box[2],box[3]
        #drawing the box
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        #print name and confidence level
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

while True:
    # will give us the image (and tell us if it was successful or not)
    success, img = cap.read()

    # run forward pass on our network using our webcam image
    # inputting out image from webcam to network(cannot use the plain image-convert image to blob)
    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    #all the available layers
    layerNames = net.getLayerNames()
    #to extract only the output layers
    #loops
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    #to send the image as a forward pass to the network and get the output from the above layers
    outputs = net.forward(outputNames)
    # print(len(outputs))
    # print(type(outputs[0]))

    #like a matrix of 300 rows x 85 cols (300 bounding boxes)
    # print(outputs[0].shape)
    # like a matrix of 1200 rows x 85 cols(1200 bounding boxes)
    # print(outputs[1].shape)
    # like a matrix of 4800 rows x 85 cols(4800 bounding boxes)
    # print(outputs[2].shape)
    #85 - 1center x(cx),2center y(cy),3width (w),4height (h),5 confidence level,rest(probabilities of the object predictions)
    #Eg if 3=0.9 in coco.name 3rd element (object inside the box is a car)
    # print(outputs[0][0])

    #go through all the boxes of 4800,1200,300 and see if the probability is good enough
    #if it is keep it and plot it or else remove it
    # go inside the output array and extract box info,probability info,object id
    findObjects(outputs,img)


    # outputing the bounding boxes

    # Window name, and the image we want to display
    cv2.imshow('Image', img)
    # the time we want to delay it (by running code up to here, the camera will be turned on)
    cv2.waitKey(1)
