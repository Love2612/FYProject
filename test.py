from ctypes.util import find_library
print(find_library('opus'))

import os
file_path = "static/processed/encoded_audio.opus"
if os.path.exists(file_path):
    print(f"File size: {os.path.getsize(file_path)} bytes")
else:
    print("File does not exist")