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
split_file = "split.txt"

def parse_annotation(file_path):
    """Extract license plate corner coordinates from annotation file."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith("corners:"):
            coords = line.replace("corners:", "").strip().split()
            coords = [list(map(int, pt.split(","))) for pt in coords]
            if len(coords) == 4:  # Ensure we have 4 points
                return coords
            else:
                return None  # Invalid annotation format
    
    return None  # No valid annotation found


def crop_license_plate(image, corners):
    """Crop and rectify the license plate using four corner points."""
    if len(corners) != 4:
        return None  # Ensure exactly 4 points

    # Convert list to NumPy array (correct shape)
    src_pts = np.array(corners, dtype=np.float32)

    # Calculate width and height for perspective transformation
    width = int(max(np.linalg.norm(src_pts[1] - src_pts[0]), np.linalg.norm(src_pts[2] - src_pts[3])))
    height = int(max(np.linalg.norm(src_pts[3] - src_pts[0]), np.linalg.norm(src_pts[2] - src_pts[1])))

    # Destination points for rectification
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Apply perspective transformation
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    lp_cropped = cv2.warpPerspective(image, M, (width, height))

    return lp_cropped

def add_gaussian_noise(image, mean=0, stddev=5):
    """Add slight Gaussian noise to an image to simulate realistic degradation."""
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def generate_lr_image(hr_image, target_ssim=0.1, max_iterations=10):
    """Degrade an HR image using mild blur and noise until SSIM drops below the target threshold."""
    lr_image = hr_image.copy()
    current_ssim = 1.0
    iteration = 0
    kernel_size = 3  # Start with minimal blur

    while current_ssim > target_ssim and iteration < max_iterations:
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        lr_image = cv2.GaussianBlur(hr_image, (kernel_size, kernel_size), 0)
        # lr_image = add_gaussian_noise(blurred_image, stddev=3)  # Add slight noise
        current_ssim = ssim(hr_image, lr_image, channel_axis=-1)
        kernel_size += 2  # Increase blur more gradually
        iteration += 1

    return lr_image

def resize_and_pad(image, target_size=(90, 256)):
    """Resize and pad an image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    pad_top = (target_size[0] - new_h) // 2
    pad_left = (target_size[1] - new_w) // 2
    
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded

def process_dataset(image_folder, hr_folder, lr_folder):
    """Process dataset to extract license plates, generate HR & LR images."""
    os.makedirs(hr_folder, exist_ok=True)
    os.makedirs(lr_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):  # Only process image files
            image_path = os.path.join(image_folder, filename)
            annotation_path = image_path.replace(".jpg", ".txt")

            if not os.path.exists(annotation_path):
                print(f"Annotation missing for {filename}, skipping...")
                continue
            print(1)
            
            # Load image and annotation
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            corners = parse_annotation(annotation_path)
            if corners is None:
                print(f"Invalid annotation format in {annotation_path}, skipping...")
                continue
            print(corners)
            
            # Step 1: Crop License Plate
            lp_cropped = crop_license_plate(image, corners)
            if lp_cropped is None:
                print(f"Error processing {filename}: Incorrect corner points.")
                continue
            print(3)
            # Step 2: Generate HR image
            lp_hr = lp_cropped

            # Step 3: Generate LR image
            lp_lr = generate_lr_image(lp_hr)
            print(4)

            lr_image = resize_and_pad(lp_lr)
            hr_image = resize_and_pad(lp_hr)

            # Step 4: Save HR and LR images
            base_name = filename.replace(".jpg", "")
            cv2.imwrite(os.path.join(hr_folder, f"{base_name}_hr.jpg"), hr_image)
            cv2.imwrite(os.path.join(lr_folder, f"{base_name}_lr.jpg"), lr_image)

            print(f"Processed: {filename}")
            with open(split_file, "a") as f:
                f.write("{};{};testing\n".format(
                    os.path.join(hr_folder, base_name + "_hr.jpg"),
                    os.path.join(lr_folder, base_name + "_lr.jpg")
                ))

# --- RUN SCRIPT ---
process_dataset(IMAGE_FOLDER, HR_FOLDER, LR_FOLDER)
