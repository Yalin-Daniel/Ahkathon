from ultralytics import YOLOE
import cv2
import numpy as np


def analyze_video_for_person_stats(video_path, model_path="best.pt", sample_frames=30):
    model = YOLOE("yoloe-11l-seg-pf.pt")
    model.predict()
    cap = cv2.VideoCapture("video/temp2.mp4")
    print("🚀 מתחיל ניתוח וידאו...")

    if not cap.isOpened():
        print("⛔ לא ניתן לפתוח את הסרטון")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height

    print(f"\n🎞️ רזולוציית הווידאו: {width}x{height} ({frame_area} פיקסלים סה\"כ)")

    person_areas = []
    person_heights = []
    person_widths = []

    frames_processed = 0
    while frames_processed < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls].lower()

            if label in ["person", "pedestrian"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                person_areas.append(area)
                person_widths.append(w)
                person_heights.append(h)

        frames_processed += 1

    cap.release()

    if not person_areas:
        print("⚠️ לא זוהו אנשים ב־30 הפריימים הראשונים")
        return

    # חישובים סטטיסטיים
    avg_area = np.mean(person_areas)
    min_area = np.min(person_areas)
    max_area = np.max(person_areas)

    avg_ratio = avg_area / frame_area

    print(f"\n🧍‍♂️ שטח ממוצע של אדם: {int(avg_area)} פיקסלים")
    print(f"📉 תיבה הכי קטנה: {min_area:.0f}, 📈 הכי גדולה: {max_area:.0f}")
    print(f"📐 יחס מהפריים (ממוצע): {avg_ratio:.5f}")

    avg_width = np.mean(person_widths)
    avg_height = np.mean(person_heights)
    print(f"\n📏 ממוצע גובה אדם בפריים: {avg_height:.1f} פיקסלים")
    print(f"📏 ממוצע רוחב אדם בפריים: {avg_width:.1f} פיקסלים")

    return {
        "resolution": (width, height),
        "frame_area": frame_area,
        "avg_area": avg_area,
        "min_area": min_area,
        "max_area": max_area,
        "avg_ratio": avg_ratio,
        "avg_height": avg_height,
        "avg_width": avg_width
    }

if __name__ == "__main__":
    analyze_video_for_person_stats("C:/Users/USER/Desktop/Ahkathon/video/temp2.mp4", model_path="best.pt")
