import cv2
import numpy as np

IMG_HEIGHT = 64
IMG_WIDTH = 1024

def load_and_preprocess_image(path_or_bytes):
    # If str → path, else → raw bytes (uploaded file)
    if isinstance(path_or_bytes, str):
        img = cv2.imread(path_or_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, -1)
    return img
