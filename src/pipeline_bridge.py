import cv2
import numpy as np
import subprocess
import os
import time

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Absolute Paths
video_480_path = 'C:/Users/sandb/Desktop/iSR_Project/data/lowres/static_480.mp4'
video_1080_path = 'C:/Users/sandb/Desktop/iSR_Project/data/raw/static_1080.mp4'
raw_input_path = 'C:/Users/sandb/Desktop/iSR_Project/data/input_480p.raw'
raw_output_path = 'C:/Users/sandb/Desktop/iSR_Project/data/output_1080p.raw'
exe_path = 'C:/Users/sandb/Desktop/iSR_Project/benchmarks/fsrcnn_pipeline.exe'
# 1. Load exact matching video frames via OpenCV
cap_480 = cv2.VideoCapture(video_480_path)
cap_1080 = cv2.VideoCapture(video_1080_path)

ret_480, frame_480 = cap_480.read()
ret_1080, frame_1080 = cap_1080.read()

cap_480.release()
cap_1080.release()

if not ret_480 or not ret_1080:
    print("Error: Could not decode video frames.")
    exit()

# 2. Dump decoded frame to raw binary bytes
print("Piping frame data down to disk...")
frame_480.tofile(raw_input_path)

# 3. Call the Compiled C++ SYCL Engine to trigger Level Zero execution
print("Spawning SYCL Subprocess on iGPU...")
start_time = time.perf_counter()
subprocess.run([exe_path], check=True)
end_time = time.perf_counter()

print(f"Hardware Compute Execution Time: {(end_time - start_time)*1000:.2f} ms")

# 4. Ingest processed frame buffer back from disk
upscaled_bytes = np.fromfile(raw_output_path, dtype=np.uint8)
frame_upscaled = upscaled_bytes.reshape((1080, 1920, 3))

# 5. Measure structural quality (PSNR validation)
gpu_psnr = calculate_psnr(frame_1080, frame_upscaled)
print(f"\n--- REAL DATA RESAMPLING TEST RESULTS ---")
print(f"iGPU Active Resolution Quality: {gpu_psnr:.2f} dB")

# Clean up temporary disk buffers
if os.path.exists(raw_input_path): os.remove(raw_input_path)
if os.path.exists(raw_output_path): os.remove(raw_output_path)