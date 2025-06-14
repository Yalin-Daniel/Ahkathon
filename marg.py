import cv2
from ultralytics import YOLO
import json

def analyze_video(video_path):
    model = YOLO("yolov8m.pt")
    input_shape = 960
    conf_thresh = 0.25
    SKIP_FRAMES = 3

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⛔ לא ניתן לפתוח את הסרטון")
        return []

    frame_count = 0
    output = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        results = model.predict(frame, imgsz=input_shape, conf=conf_thresh)[0]

        person_count = 0
        confs = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
            person_count += 1
            confs.append(float(box.conf[0]))

            # ציור תיבה סביב אדם
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # הוספת טקסט של מספר אנשים
        cv2.putText(frame, f"People: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # 🎥 תצוגה חיה
        cv2.imshow("YOLOv8 Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        confidence_avg = sum(confs) / person_count if person_count > 0 else 0.0

        output.append({
            "frame_number": frame_count,
            "people_count": person_count,
            "confidence_avg": round(confidence_avg, 4)
        })

    cap.release()
    cv2.destroyAllWindows()
    return output

# ✨ דוגמה לשימוש:
if __name__ == "__main__":
    results = analyze_video("video/video1.mp4")
    with open("results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ פלט נשמר בקובץ results_summary.json")
