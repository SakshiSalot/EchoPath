from yolov8n import run_yolo
from rtdetr import run_rtdetr
import matplotlib.pyplot as plt
import os

# video_path = "data/acad_block_videos/1.mp4"
video_path = r"C:\Users\ssesi\Desktop\git\EchoPath\data\acad_block_videos\1.mp4"
ground_truth = 50

print("Running YOLOv8n...")
yolo = run_yolo(video_path, ground_truth)

print("Running RT-DETR...")
rtdetr = run_rtdetr(video_path, ground_truth)

models = ['YOLOv8n', 'RT-DETR']

# -------------------------------
# 1. FPS 
# -------------------------------
fps = [yolo["fps"], rtdetr["fps"]]

plt.figure()
plt.bar(models, fps)
plt.title("FPS Comparison (Speed)")
plt.ylabel("Frames Per Second")
plt.show()

# -------------------------------
# 2. TIME TAKEN
# -------------------------------
time_taken = [yolo["time"], rtdetr["time"]]

plt.figure()
plt.bar(models, time_taken)
plt.title("Inference Time Comparison")
plt.ylabel("Time (seconds)")
plt.show()


# -------------------------------
# 4. MODEL SIZE
# -------------------------------
yolo_size = os.path.getsize("yolov8n.pt") / (1024 * 1024)
rtdetr_size = os.path.getsize("rtdetr-l.pt") / (1024 * 1024)

sizes = [yolo_size, rtdetr_size]

plt.figure()
plt.bar(models, sizes)
plt.title("Model Size (MB)")
plt.ylabel("Size")
plt.show()

# -------------------------------
# 5. EFFICIENCY 
# -------------------------------
efficiency = [
    yolo["fps"] / yolo_size,
    rtdetr["fps"] / rtdetr_size
]

plt.figure()
plt.bar(models, efficiency)
plt.title("Efficiency (FPS per MB)")
plt.ylabel("Score")
plt.show()

# -------------------------------
# FINAL PRINT
# -------------------------------
print("\nFINAL RESULT:")
print(f"YOLOv8n -> FPS: {yolo['fps']:.2f}, Time: {yolo['time']:.2f}, Size: {yolo_size:.2f} MB")
print(f"RT-DETR -> FPS: {rtdetr['fps']:.2f}, Time: {rtdetr['time']:.2f}, Size: {rtdetr_size:.2f} MB")