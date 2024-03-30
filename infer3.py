import cv2
import numpy as np

def load_yolo(weights_files, cfg_file, names_files):
    nets = []
    classes_list = []
    output_layers_list = []
    colors_list = []

    for weights_file, names_file in zip(weights_files, names_files):
        # Load YOLO model
        net = cv2.dnn.readNet(weights_file, cfg_file)
        nets.append(net)

        # Load classes
        with open(names_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        classes_list.append(classes)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = []
        for i in net.getUnconnectedOutLayers():
            output_layers.append(layer_names[i-1])

        output_layers_list.append(output_layers)

        # Generate random colors for bounding boxes
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        colors_list.append(colors)

    return nets, classes_list, colors_list, output_layers_list

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
            if conf > 0.2:
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

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def draw_labels_iou(boxes, confs, colors, class_ids, classes, img, iou_threshold=0.2):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if boxes[i] is None:
            continue
        x_i, y_i, w_i, h_i = boxes[i]
        label_i = str(classes[class_ids[i]])
        color_i = (0,255,0)#colors[class_ids[i]]
        
        # Check IOU with other boxes of the same class
        for j in range(i + 1, len(boxes)):
            if boxes[j] is None or class_ids[j] != class_ids[i]:
                continue
            x_j, y_j, w_j, h_j = boxes[j]
            
            # Calculate IOU between the two boxes
            iou = calculate_iou((x_i, y_i, w_i, h_i), (x_j, y_j, w_j, h_j))
            
            # If IOU exceeds threshold, suppress the box with lower confidence
            if iou > iou_threshold:
                if confs[i] > confs[j]:
                    boxes[j] = None
                else:
                    boxes[i] = None
                    break  # Break inner loop since current box is suppressed
        
        # Draw bounding box if it is not suppressed
        if boxes[i] is not None:
            cv2.rectangle(img, (x_i, y_i), (x_i + w_i, y_i + h_i), color_i, 2)
            cv2.putText(img, label_i, (x_i, y_i - 5), font, 1, color_i, 1)


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.3, 0.1)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0,255,0) #colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            # Calculate center coordinates of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            # Get the width and height of the label
            (label_width, label_height), _ = cv2.getTextSize(label, font, 1, 2)
            # Calculate the position for putting the text in the center
            text_x = max(center_x - label_width // 2, 0)
            text_y = max(center_y - label_height // 2, 0)
            # Put the text in the center of the bounding box
            cv2.putText(img, label, (text_x, text_y), font, 1, color, 1)


def process_video(video_path, loaded_yolo):
    nets, classes_list, colors_list, output_layers_list = loaded_yolo
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_image(frame)
        height, width, _ = preprocessed_frame.shape
        
        combined_boxes = []
        combined_confs = []
        combined_class_ids = []
        
        for net, classes, colors, output_layers in zip(nets, classes_list, colors_list, output_layers_list):
            blob, outputs = detect_objects(preprocessed_frame, net, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            combined_boxes.extend(boxes)
            combined_confs.extend(confs)
            combined_class_ids.extend(class_ids)
        
        # max_width = 1000
        # scale_factor = min(1.0, max_width / width)
        # new_width = int(width * scale_factor)
        # new_height = int(height * scale_factor)
        # preprocessed_frame = cv2.resize(preprocessed_frame, (new_width, new_height))

        draw_labels_iou(combined_boxes, combined_confs, colors_list[0], combined_class_ids, classes_list[0], preprocessed_frame)
        
        
        cv2.imshow('Processed Video', preprocessed_frame)
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels
# enddef

def preprocess_image(image):
    # Add preprocessing steps here if needed
    return image

def process_image(image_path, loaded_yolo):
    nets, classes_list, colors_list, output_layers_list = loaded_yolo
    image, height, width, channels = load_image(image_path)

    preprocessed_image = preprocess_image(image)
    height, width, _ = preprocessed_image.shape
    
    combined_boxes = []
    combined_confs = []
    combined_class_ids = []
    
    for net, classes, colors, output_layers in zip(nets, classes_list, colors_list, output_layers_list):
        blob, outputs = detect_objects(preprocessed_image, net, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        combined_boxes.extend(boxes)
        combined_confs.extend(confs)
        combined_class_ids.extend(class_ids)
    
    draw_labels_iou(combined_boxes, combined_confs, colors_list[0], combined_class_ids, classes_list[0], preprocessed_image)
    cv2.imshow("Image", preprocessed_image)

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
        # endif
    # endwhile
    
    return preprocessed_image

def webcam_detect(loaded_yolo):
    nets, classes_list, colors_list, output_layers_list = loaded_yolo
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_image(frame)
        height, width, _ = processed_frame.shape
        
        combined_boxes = []
        combined_confs = []
        combined_class_ids = []
        
        for net, classes, colors, output_layers in zip(nets, classes_list, colors_list, output_layers_list):
            blob, outputs = detect_objects(processed_frame, net, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            combined_boxes.extend(boxes)
            combined_confs.extend(confs)
            combined_class_ids.extend(class_ids)
        
        draw_labels(combined_boxes, combined_confs, colors_list[0], combined_class_ids, classes_list[0], processed_frame)
        
        cv2.imshow('Webcam Detection', processed_frame)
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    weights_files = [
        "custom_data_second/colab_weights/yolov3-custom-second_last (8).weights",
        "custom_data_second/colab_weights/yolov3-custom-second_last (9).weights",
        # "custom_data_second/colab_weights/yolov3-custom-second_last (10).weights",
        # "custom_data_second/colab_weights/yolov3-custom-second_last (11).weights",
        "custom_data_second/colab_weights/yolov3-custom-second_last (12).weights"
    ]
    cfg_file = "custom_data_second/yolov3-custom-second.cfg"
    names_files = [
        "custom_data_second/classes.names",
        "custom_data_second/classes.names"
    ]

    loaded_yolo = load_yolo(weights_files, cfg_file, names_files)

    # webcam_detect(loaded_yolo)

    video_path = "custom_data_second/test_videos/Pavement.MP4"
    # video_path = "custom_data_second/test_videos/GX010085_North.mp4"
    # video_path = "custom_data_second/test_videos/GX010012.MP4"
    # process_video(video_path, loaded_yolo)

    # process_image("crack.jpeg", loaded_yolo)
    process_image("custom_data_second/valid/asphalt-patch2-e1582666915314_jpg.rf.63f5bd967fe794f6a03ba3487ae3c5d8.jpg", loaded_yolo)
    # process_image("custom_data/WI48-0141--97-_jpg.rf.17173b250477a5b0f0b00037d237fa9d.jpg", loaded_yolo)
    # process_image('custom_data_second/train/us14-177-_jpg.rf.28a3a4c322feee2aec1908d69a3f0f23.jpg', loaded_yolo)
    # process_image("custom_data_second/valid/2_jpg.rf.bc00de1bad2263e0aac28e62d3037c85.jpg", loaded_yolo)
    # process_image("custom_data/US169-0-440--354-_jpg.rf.470cfb2baa2e3d66f6cac8ec5c5271a7.jpg", loaded_yolo) # to show
    # process_image("custom_data/WI48-0141--99-_jpg.rf.52fabe4a75de10be3d35b9fb9d40c4a8.jpg", loaded_yolo)
    # process_image("custom_data/WI48-0141--142-_jpg.rf.16e6801b31b0212df8707ad12ab4709c.jpg", loaded_yolo)
    # process_image("custom_data/US63_2-861-904--61-_jpg.rf.fcf621f86d307afba46dfb0e731d796d.jpg", loaded_yolo)
    # process_image("custom_data_second/multi-crack.png", loaded_yolo)
    # process_image("custom_data_second/transverse.jpeg", loaded_yolo)

if __name__ == "__main__":
    main()
