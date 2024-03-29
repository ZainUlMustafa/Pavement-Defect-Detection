import cv2
import numpy as np
import time

def load_yolo(weights_file, cfg_file, names_file):
    net = cv2.dnn.readNet(weights_file, cfg_file)
    classes = []
    output_layers = []
    with open(names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    for i in net.getUnconnectedOutLayers():
        output_layers.append(layers_names[i-1])
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf >= 0.1:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.1, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0,255,0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 3, color, 3)

def preprocess_image(image):
    return image

def process_video(video_path, weights_files, cfg_file, names_file):
    nets = []
    classes = []
    colors = []
    output_layers = []
    
    for weights_file in weights_files:
        net, cls, clr, output_layer = load_yolo(weights_file, cfg_file, names_file)
        nets.append(net)
        classes.append(cls)
        colors.append(clr)
        output_layers.append(output_layer)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_image(frame)
        height, width, channels = preprocessed_frame.shape
        
        for i in range(len(nets)):
            blob, outputs = detect_objects(preprocessed_frame, nets[i], output_layers[i])
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors[i], class_ids, classes[i], preprocessed_frame)
        
        cv2.imshow('Processed Video', preprocessed_frame)
        time.sleep(0.2)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "custom_data_second/Pavement.MP4"
    weights_files = [
        # "custom_data_second/colab_weights/yolov3-custom-second_last (9).weights", 
        # "custom_data_second/colab_weights/yolov3-custom-second_last (9).weights",
        "custom_data_second/colab_weights/yolov3-custom-second_last (11).weights",
        "custom_data_second/colab_weights/yolov3-custom-second_last (9).weights"
    ]
    cfg_file = "custom_data_second/yolov3-custom-second.cfg"
    names_file = "custom_data_second/classes.names"
    process_video(video_path, weights_files, cfg_file, names_file)

if __name__ == "__main__":
    main()
