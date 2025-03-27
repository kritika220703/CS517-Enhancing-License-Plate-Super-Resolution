import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

# --- CONFIG ---
IMAGE_FOLDER = "E:/Downloads/RodoSol-ALPR/images/dummy"  # Input folder
HR_FOLDER = "D:/downloads/DIP/mini-proj/lpsr-lacd/images/hr_new/"  # HR images
LR_FOLDER = "D:/downloads/DIP/mini-proj/lpsr-lacd/images/lr_new/"  # LR images
SSIM_THRESHOLD = 0.1  # Target degradation level for LR images
split_file = "D:/downloads/DIP/mini-proj/lpsr-lacd/split_new.txt"

def process_dataset(image_folder, hr_folder, lr_folder):
    """Process dataset to extract license plates, generate HR & LR images."""
    os.makedirs(hr_folder, exist_ok=True)
    os.makedirs(lr_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):  # Only process image files                    
            base_name = filename.replace(".jpg", "")

            print(f"Processed: {filename}")
            with open(split_file, "a") as f:
                f.write("{};{};testing\n".format(
                    os.path.join(hr_folder, base_name + "_hr.jpg"),
                    os.path.join(lr_folder, base_name + "_lr.jpg")
                ))
                
# --- RUN SCRIPT ---
process_dataset(IMAGE_FOLDER, HR_FOLDER, LR_FOLDER)