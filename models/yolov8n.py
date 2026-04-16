from ultralytics import YOLO
import time
import cv2

def run_yolo(video_path, ground_truth=50):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    total_detections = 0
    frame_count = 0

    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            total_detections += len(r.boxes)

        frame_count += 1

        cv2.imshow("YOLOv8n Detection", results[0].plot())

        if cv2.waitKey(1) & 0xFF == 27:
            break

    end = time.time()

    cap.release()
    cv2.destroyAllWindows()

    time_taken = end - start
    fps = frame_count / time_taken if time_taken > 0 else 0
    recall = total_detections / ground_truth if ground_truth > 0 else 0

    return {
        "model": "YOLOv8n",
        "time": time_taken,
        "fps": fps,
        "detections": total_detections,
        "recall": recall
    }


if __name__ == "__main__":
    video_path = "data/acad_block_videos/1.mp4"
    print(run_yolo(video_path))