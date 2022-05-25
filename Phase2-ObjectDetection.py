#!/usr/bin/env python
# coding: utf-8

# # Phase 2 - YOLO object detection

# ## Imports

# In[1]:


import numpy as np
import time
import cv2 
import os
import glob
import sys
import matplotlib.pyplot as plt
COLOR = (170,205,102)
COLOR_TEXT = (255,0,255)


# ## Passing weights and configuration file to neural network

# In[2]:


# weights_path = os.path.join("Downloads\yolo", "yolov3.weights")
# config_path = os.path.join("Downloads\yolo", "yolov3.cfg")
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
label_path = "coco.names"
labels = open(label_path).read().strip().split("\n") # strip btremove el spaces ely mwgoda


# In[3]:


net = cv2.dnn.readNetFromDarknet(config_path, weights_path)


# In[4]:


names = net.getLayerNames()
# extracting the layer names
layer_names =  net.getUnconnectedOutLayersNames()


# In[5]:


def detect(img):
   
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    (H, W) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), crop=False, swapRB = False) # swapRB = False lw ana swapthom fo2 w ana bimport
    net.setInput(blob)
    
    
    layers_output = net.forward(layer_names) # performing a forward pass through the layers

    boxes = []
    classIDs = []
    confidences = []

    for output in layers_output:
       
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)

            confidence = scores[classID]

            if (confidence > 0.85):
                box = detection[:4]*np.array([W, H, W, H]) # leh 3mlna multiply????
                bx, by, bw, bh = box.astype("int")

                # 3ayzeen el upperleft corners msh el centers 3shan kdh 3mlna el subtraction dh
                x = int(bx - bw/2)
                y = int(by - bh/2)

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = np.asarray(cv2.dnn.NMSBoxes(boxes, confidences, 0.65,0.7)) # lazem confidences tb2a float

    for i in idxs.flatten():
        (x, y, w, h) = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]

        # drawing the boxes
        img = cv2.rectangle(img, (x,y), (x+w,y+h), COLOR, 2 )

        # Putting name of detected object and accuracy of detection
        cv2.putText(img,"{} : {:.3f}".format(labels[classIDs[i]], confidences[i]), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, COLOR_TEXT,2)
    return img


# In[6]:
isVideo = int(sys.argv[1])
input_path = str(sys.argv[2])
output_path = str(sys.argv[3])


print("Video Mode: " + str(isVideo))
print("Input Path: " +input_path)
print("Output Path: " + output_path)


input_name = "project_video"

cap = cv2.VideoCapture(input_path)


output_name = input_name + '_output'
size = (1280 , 720)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(output_path + output_name +'.mp4', fourcc, 25, size)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        image = detect(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isVideo:
            
            out.write(image)
    #         cv2.imshow('frame',image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.imshow('Modified Image', image)
            cv2.waitKey(0)
            break
        
    
    else:
        break
cap.release()

out.release()

cv2.destroyAllWindows()

