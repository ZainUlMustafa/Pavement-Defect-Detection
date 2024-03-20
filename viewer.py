import os
import cv2
import pyperclip

# Global variables to keep track of current image index and list of image files
current_index = 0
image_files = []
window_name = "Image"  # Name of the OpenCV window

# Dictionary to map class IDs to their names
class_names = {
    0: "alligator_crack",
    1: "block_crack",
    2: "longitudinal_crack",
    3: "pothole",
    4: "sealed_longitudinal_crack",
    5: "sealed_transverse_crack",
    6: "transverse_crack"
}

# Function to read annotations from a text file
def read_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x, y, width, height = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append((class_names[class_id], x, y, width, height))
    return annotations

# Function to display image with annotations using OpenCV
def display_image_with_annotations():
    global current_index, image_files
    
    # Open the image
    image_file = image_files[current_index]
    annotation_file = image_file.replace(".jpg", ".txt")
    image = cv2.imread(image_file)
    
    # Read annotations
    annotations = read_annotations(annotation_file)
    
    # Draw annotations
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
    
    # Display the image with name
    image_name = os.path.basename(image_file)
    cv2.putText(image, f'{current_index+1}/{len(image_files)} {image_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 5, 0), 1, cv2.LINE_AA)
    
    # Show the image in the existing window or create a new one
    cv2.imshow(window_name, image)

    # Copy the image name without extension to the clipboard
    pyperclip.copy(os.path.splitext(image_name)[0])

# Function to display the next image
def next_image():
    global current_index
    current_index = (current_index + 1) % len(image_files)
    display_image_with_annotations()

# Function to display the previous image
def prev_image():
    global current_index
    current_index = (current_index - 1) % len(image_files)
    display_image_with_annotations()

# Main function to initialize image files list and display the first image
def main():
    global image_files
    
    # Populate list of image files
    folder_path = "./custom_data"  # Update the folder path
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            image_files.append(os.path.join(folder_path, file))
    
    # Display the first image
    if image_files:
        display_image_with_annotations()
    
    # Listen for key events to navigate between images
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('n'):
            next_image()
        elif key == ord('p'):
            prev_image()

# Call the main function
if __name__ == "__main__":
    main()

# Close OpenCV windows
cv2.destroyAllWindows()
