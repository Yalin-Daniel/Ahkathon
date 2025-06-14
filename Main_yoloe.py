import cv2
import numpy as np
import os
from ultralytics import YOLO

# ⚙️ טען את מודל YOLOv8x (מדויק אך כבד – מתאים לרחפן)
model = YOLO("yolov8x.pt")

input_shape = 1280  # רזולוציה גבוהה יותר לאנשים מרחוק
conf_thresh = 0.25
SKIP_FRAMES = 3

# 🗂️ ודא שתיקיית הפלט קיימת
os.makedirs("results_json", exist_ok=True)

# 🎥 פתח את הסרטון
cap = cv2.VideoCapture("video/video1.mp4")
if not cap.isOpened():
    print("⛔ לא ניתן לפתוח את הסרטון"); exit()

frame_count = 0
print("🎥 זיהוי התחיל, לחץ 'q' כדי לצאת")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    # 🧠 הרצת המודל
    results = model.predict(frame, imgsz=input_shape, conf=conf_thresh)[0]

    person_count = 0

    # 📦 ציור תיבות רק על אנשים
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue  # מזהה רק אנשים

        person_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = f"Person {conf:.2f}"

        # 💚 ריבוע ירוק סביב אדם
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 🧮 כתוב את כמות האנשים על המסך
    cv2.putText(frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # 📝 שמירת JSON
    with open(f"results_json/frame_{frame_count:05d}.json", "w") as f:
        f.write(results.to_json())

    # 🖼️ תצוגה חיה
    cv2.imshow("YOLOv8 – Drone Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ סיום")