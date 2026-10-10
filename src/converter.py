import cv2
import os

# Define paths
raw_dir = 'C:/Users/sandb/Desktop/iSR_Project/data/raw'
lowres_dir = 'C:/Users/sandb/Desktop/iSR_Project/data/lowres'

# Ensure the output directory exists
if not os.path.exists(lowres_dir):
    os.makedirs(lowres_dir)

def create_lowres(filename):
    input_path = os.path.join(raw_dir, filename)
    output_path = os.path.join(lowres_dir, filename.replace('1080', '480'))
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not find {input_path}")
        return

    # Get original properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # We are forcing the output to 854x480 (standard 16:9 480p)
    out = cv2.VideoWriter(output_path, fourcc, fps, (854, 480))
    
    print(f"Processing {filename}...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # This is the 'Dumb' downscale
        low_res_frame = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_AREA)
        out.write(low_res_frame)
        
    cap.release()
    out.release()
    print(f"Successfully created: {output_path}")

# Run for both your files
create_lowres('static_1080.mp4') # Make sure the extension (.mp4) matches exactly!
create_lowres('dynamic_1080.mp4')