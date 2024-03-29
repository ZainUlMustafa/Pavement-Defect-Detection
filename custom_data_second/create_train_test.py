import os

# Function to get all image filenames with .jpg extension
def get_image_filenames(folder_path):
    image_filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_filenames.append(os.path.join(root, file))
    return image_filenames

# Function to write filenames to data.txt
def write_to_data_txt(image_filenames, output_file, root_folder):
    with open(output_file, 'w') as f:
        for filename in image_filenames:
            f.write(f"{root_folder}/test/{os.path.basename(filename)}\n")

# Main function
def main():
    # Specify the folder path containing the images
    root_folder = "custom_data_second"
    folder_path = "./valid"  # Update with your folder path
    
    # Get image filenames
    image_filenames = get_image_filenames( folder_path)
    
    # Write filenames to data.txt
    output_file = "test.txt"
    write_to_data_txt(image_filenames, output_file, root_folder)
    print(f"File names written to {output_file}")

if __name__ == "__main__":
    main()
