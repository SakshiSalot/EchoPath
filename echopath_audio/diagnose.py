import cv2
import easyocr
import numpy as np

INPUT_VIDEO = "echopath_Persons.mp4"   # ← your video name

reader = easyocr.Reader(['en'], gpu=False)
cap    = cv2.VideoCapture(INPUT_VIDEO)
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {total} frames @ {fps} FPS")
print("="*60)

# Sample 10 frames spread across the video
sample_frames = [int(total * i / 10) for i in range(1, 10)]

for fn in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        continue

    print(f"\n--- Frame {fn} ({fn/fps:.1f}s) ---")
    results = reader.readtext(frame)

    if not results:
        print("  [nothing detected]")
        continue

    for (_, text, conf) in results:
        print(f"  conf={conf:.2f}  '{text}'")

    # Also save the frame so you can visually check it
    cv2.imwrite(f"frame_{fn}.jpg", frame)
    print(f"  → Saved frame_{fn}.jpg")

cap.release()
print("\nDone. Check the saved .jpg files to see if bounding boxes are visible.")