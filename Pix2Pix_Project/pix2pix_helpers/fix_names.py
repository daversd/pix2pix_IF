import os

"""Use this script to fix the name of files in folders
Note: Backup your files before proceeding.
"""

FOLDER = os.path.realpath("sample_dataset")
TARGET = "level "


for (dirpath, dirnames, filenames) in os.walk(FOLDER):
    for filename in filenames:
        x = os.path.basename(dirpath) + '_'
        if TARGET in filename:
            renamed = filename.replace(TARGET, x)
            os.rename(os.path.join(dirpath, filename),
                      os.path.join(dirpath, renamed))
