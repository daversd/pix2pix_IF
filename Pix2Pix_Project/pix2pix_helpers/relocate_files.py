import os
import shutil

"""Use this script to move files to a folder
"""

# Set the paths
SOURCE = os.path.realpath("sample_dataset/AB")
TARGET = os.path.realpath("sample_dataset/combined")

# Create the target directory if it doesn't exist
if not os.path.isdir(TARGET):
    os.makedirs(TARGET)

# Move the files
for (dirpath, dirnames, filenames) in os.walk(SOURCE):
    for filename in filenames:
        srcPath = os.path.join(dirpath, filename)
        toPath = os.path.join(TARGET, filename)
        if os.path.isfile(toPath):
            print(f"File {toPath} exists! Overwriting has been avoided")
        else:
            shutil.move(srcPath, toPath)
