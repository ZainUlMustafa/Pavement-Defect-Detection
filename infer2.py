import cv2
import numpy as np
import argparse
import time


def load_yolo():
    net = cv2.dnn.readNet("custom_data_second/colab_weights/yolov3-custom-second_last (8).weights", "custom_data_second/yolov3-custom-second.cfg")
    # net = cv2.dnn.readNet("custom_weight/yolov3_608.weights", "cfg/yolov3_608.cfg")
    classes = []
    output_layers = []
    with open("custom_data_second/classes.names", "r") as f:
    # with open("labels/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    # print(layers_names)
    for i in net.getUnconnectedOutLayers():
        output_layers.append(layers_names[i-1])
    # endfor
    # output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers
# enddef


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels
# enddef


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs
# enddef


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            # print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf >= 0.1:
                print(conf)
                # print(class_id)
                print(class_id, conf, detect)
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
            # endif
        # endfor
    # endfor
    return boxes, confs, class_ids
# enddef


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.1, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    print(indexes)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0,255,0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 3, color, 3)
        # endif
    # endfor
    cv2.imshow("Image", img)
# enddef


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    # print(class_ids)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
        # endif
    # endwhile
# enddef


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    # cap = start_webcam()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        # endif
    # endwhile
    cap.release()
# enddef
    
def process_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        
        cv2.imshow('Processed Video', frame)
        time.sleep(0.2)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
#enddef


def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        # endif
    # endwhile
    cap.release()
# enddef


def main():
    print("Main")
    process_video("custom_data_second/Pavement.MP4")
    # webcam_detect()
    # image_detect("indoor.jpeg")
    # image_detect("books.jpg")
    # image_detect("crack.jpeg")
    # image_detect("custom_data_second/valid/asphalt-patch2-e1582666915314_jpg.rf.63f5bd967fe794f6a03ba3487ae3c5d8.jpg")
    # image_detect("custom_data/WI48-0141--97-_jpg.rf.17173b250477a5b0f0b00037d237fa9d.jpg")
    # image_detect('custom_data_second/train/us14-177-_jpg.rf.28a3a4c322feee2aec1908d69a3f0f23.jpg')
    # image_detect("custom_data_second/valid/2_jpg.rf.bc00de1bad2263e0aac28e62d3037c85.jpg")
    # image_detect("custom_data/US169-0-440--354-_jpg.rf.470cfb2baa2e3d66f6cac8ec5c5271a7.jpg") # to show
    # image_detect("custom_data/WI48-0141--99-_jpg.rf.52fabe4a75de10be3d35b9fb9d40c4a8.jpg")
    # image_detect("custom_data/WI48-0141--142-_jpg.rf.16e6801b31b0212df8707ad12ab4709c.jpg")
    # image_detect("custom_data/US63_2-861-904--61-_jpg.rf.fcf621f86d307afba46dfb0e731d796d.jpg")
# enddef


if __name__ == "__main__":
    main()
# endif