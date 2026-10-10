import cv2
import time
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Paths - Update filenames if yours are different
video_480_path = 'C:/Users/sandb/Desktop/iSR_Project/data/lowres/static_480.mp4'
video_1080_path = 'C:/Users/sandb/Desktop/iSR_Project/data/raw/static_1080.mp4'

cap_480 = cv2.VideoCapture(video_480_path)
cap_1080 = cv2.VideoCapture(video_1080_path)

fps_list = []
psnr_list = []

print("Running Baseline (Bicubic) Benchmark...")

while cap_480.isOpened() and cap_1080.isOpened():
    ret_low, frame_low = cap_480.read()
    ret_high, frame_high = cap_1080.read()
    
    if not ret_low or not ret_high:
        break
    
    # 1. Measure Throughput (Time the upscale only)
    start = time.perf_counter()
    # Standard Bicubic Upscale
    upscaled = cv2.resize(frame_low, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    end = time.perf_counter()
    
    fps_list.append(1 / (end - start))
    
    # 2. Measure Quality (PSNR)
    # Ensure frames are the same size for comparison
    psnr_val = calculate_psnr(frame_high, upscaled)
    psnr_list.append(psnr_val)

cap_480.release()
cap_1080.release()

print(f"\n--- BASELINE RESULTS (STATIC CLIP) ---")
print(f"Average Throughput: {np.mean(fps_list):.2f} FPS")
print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")