import os

def read_empty_files(empty_files_file):
    empty_files = set()
    with open(empty_files_file, 'r') as f:
        for line in f:
            empty_files.add(line.strip())
    return empty_files

def remove_empty_files_from_train(train_file, empty_files):
    lines = []
    with open(train_file, 'r') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        filename = line.strip()
        if not any(empty_file in filename for empty_file in empty_files):
            filtered_lines.append(line)

    with open(train_file, 'w') as f:
        f.writelines(filtered_lines)

# Paths to the files
empty_files_file = "faulty-images.txt"
train_file = "./custom_data/test.txt"

# Read empty files
empty_files = read_empty_files(empty_files_file)

# Remove empty file entries from train file
remove_empty_files_from_train(train_file, empty_files)

print("Empty files removed from train file successfully.")
