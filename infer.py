import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("backup/yolov3-custom_last.weights", "custom_data/yolov3-custom.cfg")
layer_names = net.getLayerNames()
print(layer_names)
output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i-1])

# Load classes
classes = []
with open("custom_data/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
img = cv2.imread("books.jpg")
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence >= 0.0:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
print(boxes[0])
for i in range(len(boxes)):
    if i in indexes:
        print(boxes[i])
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (min(x + w, img.shape[1]), min(y + h, img.shape[0])), color, 2)
        # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 30), font, 3, color, 3)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
