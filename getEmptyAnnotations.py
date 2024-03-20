import os

def find_empty_txt_files(folder_path):
    empty_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            if os.path.getsize(file_path) == 0:
                empty_files.append(os.path.splitext(file)[0])
    return empty_files

def save_to_txt(file_names, output_file):
    with open(output_file, "w") as f:
        for file_name in file_names:
            f.write(file_name + "\n")

# Example usage:
folder_path = "./custom_data"  # Replace with the path to your folder
output_file = "faulty-images.txt"  # Name of the output file

empty_files = find_empty_txt_files(folder_path)
if empty_files:
    print("Empty text files found:")
    for file_name in empty_files:
        print(file_name)
    save_to_txt(empty_files, output_file)
    print("List of empty files saved to", output_file)
else:
    print("No empty text files found in the folder.")
