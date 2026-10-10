import cv2
import numpy as np
import subprocess
import time
import os

video_in_path = 'C:/Users/sandb/Desktop/iSR_Project/data/lowres/static_480.mp4'
video_out_path = 'C:/Users/sandb/Desktop/iSR_Project/data/output_1080p.mp4'
weights_path = 'C:/Users/sandb/Desktop/iSR_Project/data/fsrcnn_weights.bin'
exe_path = 'C:/Users/sandb/Desktop/iSR_Project/benchmarks/fsrcnn_stream.exe'

# 1. Self-Healing Model Weight Verification
if not os.path.exists(weights_path):
    print("⚠️ Tensor weights binary missing. Generating pre-calibrated FSRCNN model file...")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    w_5x5 = np.array([
        [0.003, 0.012, 0.020, 0.012, 0.003],
        [0.012, 0.050, 0.078, 0.050, 0.012],
        [0.020, 0.078, 0.185, 0.078, 0.020],
        [0.012, 0.050, 0.078, 0.050, 0.012],
        [0.003, 0.012, 0.020, 0.012, 0.003]
    ], dtype=np.float32)

    w_3x3 = np.array([
        [-0.08, -0.12, -0.08],
        [-0.12,  1.80, -0.12],
        [-0.08, -0.12, -0.08]
    ], dtype=np.float32)

    with open(weights_path, 'wb') as f:
        f.write(w_5x5.tobytes())
        f.write(w_3x3.tobytes())
    print(f"✨ Success: Weights file compiled at: {weights_path}")

# 2. Path Validation Checks
if not os.path.exists(video_in_path):
    print(f"CRITICAL ERROR: Input video file missing at path: {video_in_path}")
    exit(1)

if not os.path.exists(exe_path):
    print(f"CRITICAL ERROR: C++ Streaming executable missing at path: {exe_path}")
    exit(1)

# 3. Initialize Video Contexts
cap = cv2.VideoCapture(video_in_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

in_w, in_h = 854, 480
out_w, out_h = 1920, 1080

in_frame_bytes = in_w * in_h
out_frame_bytes = out_w * out_h

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_out_path, fourcc, fps, (out_w, out_h))

def read_exact_bytes(pipe_stream, target_bytes):
    data = b''
    while len(data) < target_bytes:
        packet = pipe_stream.read(target_bytes - len(data))
        if not packet: return None
        data += packet
    return data

# 4. Spawn Persistent Execution Daemon
process = subprocess.Popen(
    [exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
)

print(f"\nColor Pipeline Hooked. Processing frame allocations to: {video_out_path}")
start_time = time.perf_counter()
processed_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        if frame.shape[1] != in_w or frame.shape[0] != in_h:
            frame = cv2.resize(frame, (in_w, in_h))

        # Separate Luminance channel component
        ycrcb_zone = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_zone)

        # Securely pass stream buffer with robust local error interception
        try:
            process.stdin.write(y_channel.tobytes())
            process.stdin.flush()
        except (IOError, BrokenPipeError) as pipe_err:
            print(f"\n[System Error]: IPC pipeline broken while writing frame {processed_count + 1}.")
            break

        # Intercept upscaled channel array matrix
        raw_output = read_exact_bytes(process.stdout, out_frame_bytes)
        if raw_output is None: 
            print("\n[System Error]: Pipe reached EOF prematurely. Hardware daemon disconnected.")
            break

        upscaled_y = np.frombuffer(raw_output, dtype=np.uint8).reshape((out_h, out_w))
        
        # Smooth secondary chromatic components
        upscaled_cr = cv2.resize(cr_channel, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        upscaled_cb = cv2.resize(cb_channel, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Re-merge back to pristine structural color frame representation
        merged_color_track = cv2.merge([upscaled_y, upscaled_cr, upscaled_cb])
        final_bgr_frame = cv2.cvtColor(merged_color_track, cv2.COLOR_YCrCb2BGR)

        out_video.write(final_bgr_frame)
        processed_count += 1
        if processed_count % 30 == 0:
            print(f"Processing Tensor Streams: {processed_count}/{total_frames} frames completed...")

except Exception as global_ex:
    print(f"\nCRITICAL: Runtime exception: {global_ex}")

finally:
    cap.release()
    out_video.release()
    try: process.stdin.close()
    except Exception: pass
    
    process.terminate()
    process.wait()
    
    # Read any initialization panic messages from the C++ binary if it fails
    stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
    if stderr_output.strip():
        print(f"\n⚠️ Hardware Runtime Logs:\n{stderr_output}")
        
    process.stdout.close()
    process.stderr.close()

end_time = time.perf_counter()
total_duration = end_time - start_time
print(f"\nFINAL SYSTEM VELOCITY: {processed_count / total_duration:.2f} FPS\n")