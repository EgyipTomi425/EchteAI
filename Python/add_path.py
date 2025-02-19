import sys
import os

current_directory = os.getcwd()

for entry in os.scandir(current_directory):
    if entry.is_dir():
        sys.path.append(entry.path)

print("Current Python Path:", sys.path)
