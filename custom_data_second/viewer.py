import os
import cv2
import pyperclip

current_index = 0
image_files = []
window_name = "Image"

class_names = {
    0: "alligator crack",
    1: "edge crack",
    2: "longitudinal cracking",
    3: "patching",
    4: "rutting",
    5: "transverse cracking",
}

def read_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x, y, width, height = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append((class_names[class_id], x, y, width, height))
        #endfor
    #endwith
    return annotations
#enddef

def display_image_with_annotations():
    global current_index, image_files
    
    image_file = image_files[current_index]
    annotation_file = image_file.replace(".jpg", ".txt")
    image = cv2.imread(image_file)
    
    annotations = read_annotations(annotation_file)
    
    for annotation in annotations:
        class_name, x, y, width, height = annotation
        x *= image.shape[1]
        y *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]
        pt1 = (int(x - width/2), int(y - height/2))
        pt2 = (int(x + width/2), int(y + height/2))
        cv2.rectangle(image, pt1, pt2, (0, 0, 255), 1)
        cv2.putText(image, f'{class_name}', (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #endfor
    
    image_name = os.path.basename(image_file)
    cv2.putText(image, f'{current_index+1}/{len(image_files)} {image_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 5, 0), 1, cv2.LINE_AA)
    
    cv2.imshow(window_name, image)

    pyperclip.copy(os.path.splitext(image_name)[0])
#enddef

def next_image():
    global current_index
    current_index = (current_index + 1) % len(image_files)
    display_image_with_annotations()
#enddef


def prev_image():
    global current_index
    current_index = (current_index - 1) % len(image_files)
    display_image_with_annotations()
#enddef

def main():
    global image_files
    
    folder_path = "./train" 
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            image_files.append(os.path.join(folder_path, file))
        #endif
    #endfor
    
    if image_files:
        display_image_with_annotations()
    #endif
    
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('n'):
            next_image()
        elif key == ord('p'):
            prev_image()
        #endif
    #endwhile
#enddef


if __name__ == "__main__":
    main()
#endif

cv2.destroyAllWindows()
