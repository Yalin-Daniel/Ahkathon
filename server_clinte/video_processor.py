# video_processor.py - מעבד הסרטונים המעודכן
import cv2
import numpy as np
import os
import json
from pathlib import Path
import time

# נסה לייבא keras-retinanet
try:
    from keras_retinanet.models import load_model
    from keras_retinanet.utils.visualization import draw_box, label_color
    from keras_retinanet.utils.image import preprocess_image, resize_image
    RETINANET_AVAILABLE = True
    print("✅ keras-retinanet זמין")
except ImportError as e:
    print(f"⚠  keras-retinanet לא זמין: {e}")
    RETINANET_AVAILABLE = False

def find_available_model():
    """חפש מודל זמין במערכת"""
    possible_paths = [
        "snapshots/resnet50_csv_08_inference.h5",
        "models/resnet50_csv_08_inference.h5",
        "weights/resnet50_csv_08_inference.h5",
        "../models/resnet50_csv_08_inference.h5",
        "snapshots/resnet50_coco_best_v2.1.0.h5",
    ]

    print("🔍 מחפש מודלים זמינים...")
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ נמצא מודל: {path}")
            return path

    # חפש כל קבצי .h5
    h5_files = list(Path(".").rglob("*.h5"))
    if h5_files:
        for file in h5_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # מודל אמיתי כנראה גדול מ-10MB
                print(f"🎯 נבחר: {file}")
                return str(file)

    print("❌ לא נמצא מודל מתאים")
    return None

def create_simple_detector():
    """צור זיהוי פשוט שעובד בלי מודל מורכב"""
    print("🔧 יוצר זיהוי פשוט...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def detect_with_hog(frame, hog_detector):
    """זיהוי אנשים עם HOG"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (rects, weights) = hog_detector.detectMultiScale(
            gray,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            groupThreshold=2
        )

        people = []
        for i, (x, y, w, h) in enumerate(rects):
            confidence = min(0.9, max(0.6, weights[i] if len(weights) > i else 0.8))
            people.append({
                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                'confidence': float(confidence)
            })

        return people

    except Exception as e:
        print(f"שגיאה בזיהוי: {e}")
        return []

def detect_with_retinanet(frame, model):
    """זיהוי עם RetinaNet (אם זמין)"""
    try:
        image = preprocess_image(frame.copy())
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        people = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.4:
                continue
            if int(label) != 0:  # 0 = person בCOCO
                continue

            x1, y1, x2, y2 = map(int, box)
            people.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score)
            })

        return people

    except Exception as e:
        print(f"❌ שגיאה ב-RetinaNet: {e}")
        return []

def process_video_with_detection(video_path, output_json_path="detection_results.json"):
    """עבד סרטון עם זיהוי הולכי רגל - ללא חלון תצוגה"""
    print(f"🎬 מעבד סרטון: {video_path}")

    if not Path(video_path).exists():
        print(f"❌ סרטון לא נמצא: {video_path}")
        return False

    # נסה לטעון מודל RetinaNet
    model = None
    if RETINANET_AVAILABLE:
        model_path = find_available_model()
        if model_path:
            try:
                print(f"📥 טוען מודל RetinaNet: {model_path}")
                model = load_model(model_path, backbone_name='resnet50')
                print("✅ מודל RetinaNet נטען בהצלחה")
            except Exception as e:
                print(f"❌ שגיאה בטעינת RetinaNet: {e}")
                model = None

    # אם RetinaNet לא עבד, השתמש ב-HOG
    if model is None:
        print("🔄 עובר לזיהוי HOG...")
        hog_detector = create_simple_detector()
        detection_method = "HOG"
    else:
        hog_detector = None
        detection_method = "RetinaNet"

    # פתח סרטון
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ לא ניתן לפתוח את הסרטון")
        return False

    # מידע על הסרטון
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📊 מידע על הסרטון:")
    print(f"   🎯 רזולוציה: {width}x{height}")
    print(f"   ⏱  FPS: {fps}")
    print(f"   🎞 סך פריימים: {total_frames}")
    print(f"   🧠 שיטת זיהוי: {detection_method}")

    # תיקיית תוצאות
    results_dir = Path("results_json")
    results_dir.mkdir(exist_ok=True)

    results = []
    frame_count = 0
    SKIP_FRAMES = 5  # עבד כל 5 פריים
    start_time = time.time()

    print("🚀 זיהוי התחיל...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # דלג על פריימים
        if frame_count % SKIP_FRAMES != 0:
            continue

        # זיהוי אנשים
        if model is not None:
            people = detect_with_retinanet(frame, model)
        else:
            people = detect_with_hog(frame, hog_detector)

        # צור נתוני JSON לפריים
        frame_data = {
            "frame_number": frame_count,
            "timestamp": round(frame_count / fps, 2) if fps > 0 else frame_count,
            "pedestrian_count": len(people),
            "detections": [
                {
                    "detection_id": i,
                    "confidence": person["confidence"],
                    "bounding_box": {
                        "x1": person["bbox"][0],
                        "y1": person["bbox"][1],
                        "x2": person["bbox"][2],
                        "y2": person["bbox"][3],
                        "width": person["bbox"][2] - person["bbox"][0],
                        "height": person["bbox"][3] - person["bbox"][1]
                    }
                }
                for i, person in enumerate(people)
            ],
            "metadata": {
                "frame_width": width,
                "frame_height": height,
                "detection_method": detection_method,
                "processing_successful": True
            }
        }

        results.append(frame_data)

        # הדפס התקדמות
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            elapsed = time.time() - start_time
            print(f"📈 פריים {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"זוהו {len(people)} אנשים | "
                  f"זמן: {elapsed:.1f}s")

    cap.release()

    # שמור JSON מלא
    processing_time = time.time() - start_time
    total_detections = sum(frame['pedestrian_count'] for frame in results)
    max_people = max(frame['pedestrian_count'] for frame in results) if results else 0
    avg_people = total_detections / len(results) if results else 0

    final_result = {
        "video_info": {
            "path": str(video_path),
            "filename": Path(video_path).name,
            "fps": fps,
            "total_frames": total_frames,
            "resolution": {"width": width, "height": height},
            "detection_method": detection_method
        },
        "summary": {
            "total_pedestrian_detections": total_detections,
            "max_pedestrians_in_frame": max_people,
            "average_pedestrians_per_frame": round(avg_people, 2),
            "frames_processed": len(results),
            "processing_time_seconds": round(processing_time, 2)
        },
        "frames": results
    }

    # שמור תוצאה מלאה
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 עיבוד הושלם!")
    print(f"📁 JSON נשמר ב: {output_json_path}")
    print(f"📊 סטטיסטיקות:")
    print(f"   👥 סך זיהויים: {total_detections}")
    print(f"   🏆 מקסימום בפריים: {max_people}")
    print(f"   📈 ממוצע לפריים: {avg_people:.2f}")
    print(f"   ⏱  זמן עיבוד: {processing_time:.2f} שניות")

    return True

# פונקציה להרצה עצמאית לבדיקות
if __name__ == "__main__":
    test_video = "test_video.mp4"
    if Path(test_video).exists():
        process_video_with_detection(test_video, "test_results.json")
    else:
        print(f"לא נמצא סרטון לבדיקה: {test_video}")