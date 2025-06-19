#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pedestrian detection in drone videos - Fixed Version
עם חיבור משופר למפה הדינמית - ללא שגיאות
"""

import cv2
import numpy as np
import json
from pathlib import Path
import time
import requests
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from typing import List

# Try importing keras-retinanet
try:
    from keras_retinanet.models import load_model
    from keras_retinanet.utils.visualization import draw_box, label_color
    from keras_retinanet.utils.image import preprocess_image, resize_image

    RETINANET_AVAILABLE = True
    print("✅ keras-retinanet is available")
except ImportError as e:
    print(f"⚠️  keras-retinanet not available: {e}")
    RETINANET_AVAILABLE = False

# הגדרות API
BASE_URL = "http://localhost:8080"
VIDEO_ENDPOINT = f"{BASE_URL}/insert_video/"
LOCATION_UPDATE_ENDPOINT = f"{BASE_URL}/update_location/"
STATS_ENDPOINT = f"{BASE_URL}/get_video_stats/"

# הגדרות וידאו
video_path = "video/video1.mp4"

# הגדרות עיבוד
SKIP_FRAMES = 7
CONFIDENCE_THRESHOLD = 0.4


def check_server_connection():
    """בדיקת חיבור לשרת"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ חיבור לשרת הצליח")
            return True
        else:
            print(f"⚠️ שרת מגיב אבל עם שגיאה: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ לא ניתן להתחבר לשרת ב-{BASE_URL}")
        print("💡 ודא שהשרת רץ על localhost:8080")
        return False
    except requests.exceptions.Timeout:
        print("⏱️ זמן החיבור לשרת פג")
        return False


def get_server_stats():
    """קבלת סטטיסטיקות מהשרת"""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 סטטיסטיקות שרת:")
            print(f"   📝 סה״כ רשומות: {stats.get('total_entries', 0)}")
            print(f"   👥 סה״כ אנשים זוהו: {stats.get('total_people_detected', 0)}")
            print(f"   🎬 וידאוים ייחודיים: {stats.get('unique_videos', 0)}")
            print(f"   📍 רשומות עם מיקום: {stats.get('entries_with_location', 0)}")
            return stats
        else:
            print(f"❌ שגיאה בקבלת סטטיסטיקות: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ שגיאה בחיבור לסטטיסטיקות: {e}")
        return None


def create_video_entry1(data):
    """יצירת רשומת וידאו חדשה עם שיפורים"""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(VIDEO_ENDPOINT, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Frame {data['frame_number']}: נוסף בהצלחה (ID: {result.get('id')})")
                return True
            else:
                print(f"❌ Frame {data['frame_number']}: שגיאה {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"🔄 ניסיון חוזר {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        except requests.exceptions.ConnectionError:
            print(f"❌ Frame {data['frame_number']}: שגיאת חיבור")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
        except requests.exceptions.Timeout:
            print(f"⏱️ Frame {data['frame_number']}: זמן הבקשה פג")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            print(f"❌ Frame {data['frame_number']}: שגיאת בקשה - {e}")
            break

    print(f"💥 Frame {data['frame_number']}: נכשל לאחר {max_retries} ניסיונות")
    return False

def create_video_entry2(video_path: str, excel_path: str, model=None):
    """
    פונקציה אחת שמבצעת את כל התהליך:
    טעינת וידאו + נתוני מיקום + זיהוי אנשים + שליחה עם מיקום בפעם אחת לשרת
    """
    # פתיחת וידאו
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ לא ניתן לפתוח את הוידאו")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📽️ FPS: {fps}, Total frames: {total_frames}")

    # טען מיקום מ־Excel למילון
    try:
        df = pd.read_excel(excel_path)
        df['timestamp'] = df['timestamp'].astype(str).str.replace('s', '', regex=False).astype(float)
        df['frame_number'] = (df['timestamp'] * fps).astype(int)
        location_map = {
            int(row['frame_number']): {
                "height": float(row['altitude']),
                "longitude": float(row['longitude']),
                "latitude": float(row['latitude']),
            }
            for _, row in df.iterrows()
        }
        print(f"✅ נטענו {len(location_map)} פריימים עם מיקום")
    except Exception as e:
        print(f"❌ שגיאה בטעינת Excel: {e}")
        location_map = {}

    # הגלאי (למשל HOG)
    if model is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 7 != 0:
            continue

        # זיהוי אנשים
        if model is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
            people_count = len(rects)
        else:
            # RetinaNet (אם קיים)
            image = preprocess_image(frame.copy())
            image, scale = resize_image(image)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale
            people_count = sum(1 for score, label in zip(scores[0], labels[0]) if score > 0.4 and label == 0)

        # זמן ותזמון
        timestamp_seconds = round(frame_count / fps, 2) if fps > 0 else frame_count
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # מיקום מה-Excel
        location = location_map.get(frame_count, {
            "height": 0,
            "longitude": 0.0,
            "latitude": 0.0
        })

        payload = {
            "frame_number": frame_count,
            "timestamp": timestamp_seconds,
            "pedestrian_count": people_count,
            "time": now_time,
            "video_path": video_path,
            "height": location["height"],
            "longitude": location["longitude"],
            "latitude": location["latitude"]
        }

        try:
            res = requests.post(f"{BASE_URL}/insert_video/", json=payload)
            if res.status_code == 200:
                print(f"✅ Frame {frame_count}: נשלח עם {people_count} אנשים ומיקום ({location['latitude']}, {location['longitude']})")
            else:
                print(f"❌ שגיאה בשליחה לשרת: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ שגיאה בבקשה: {e}")

    cap.release()
    cv2.destroyAllWindows()

def create_video_entry(video_path: str, excel_path: str, model=None):
    """
    טעינת וידאו + נתוני מיקום + זיהוי אנשים + שליחה עם מיקום הקרוב ביותר לכל פריים מדוגם
    """

    def find_closest_location(frame_number, location_map, max_distance=15):
        """מאתר את מיקום ה-GPS הקרוב ביותר לפריים"""
        if not location_map:
            return {'height': 0, 'longitude': 0.0, 'latitude': 0.0}

        closest_frame = min(location_map.keys(), key=lambda x: abs(x - frame_number))
        if abs(closest_frame - frame_number) > max_distance:
            return {'height': 0, 'longitude': 0.0, 'latitude': 0.0}
        return location_map[closest_frame]

    # פתיחת וידאו
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ לא ניתן לפתוח את הוידאו")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📽️ FPS: {fps}, Total frames: {total_frames}")

    # טעינת מיקום מקובץ Excel
    try:
        df = pd.read_excel(excel_path)
        df['timestamp'] = df['timestamp'].astype(str).str.replace('s', '', regex=False).astype(float)
        df['frame_number'] = (df['timestamp'] * fps).astype(int)
        location_map = {
            int(row['frame_number']): {
                "height": float(row['altitude']),
                "longitude": float(row['longitude']),
                "latitude": float(row['latitude']),
            }
            for _, row in df.iterrows()
        }
        print(f"✅ נטענו {len(location_map)} פריימים עם מיקום")
    except Exception as e:
        print(f"❌ שגיאה בטעינת Excel: {e}")
        location_map = {}

    # גלאי ברירת מחדל אם אין מודל RetinaNet
    if model is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue  # רק כל N פריימים

        # זיהוי אנשים
        if model is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
            people_count = len(rects)
        else:
            image = preprocess_image(frame.copy())
            image, scale = resize_image(image)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale
            people_count = sum(1 for score, label in zip(scores[0], labels[0]) if score > 0.4 and label == 0)

        # זמן
        timestamp_seconds = round(frame_count / fps, 2) if fps > 0 else frame_count
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # קבלת מיקום קרוב
        location = find_closest_location(frame_count, location_map)

        payload = {
            "frame_number": frame_count,
            "timestamp": timestamp_seconds,
            "pedestrian_count": people_count,
            "time": now_time,
            "video_path": video_path,
            "height": location["height"],
            "longitude": location["longitude"],
            "latitude": location["latitude"]
        }

        try:
            res = requests.post(f"{BASE_URL}/insert_video/", json=payload)
            if res.status_code == 200:
                print(f"✅ Frame {frame_count}: נשלח עם {people_count} אנשים ומיקום ({location['latitude']}, {location['longitude']})")
            else:
                print(f"❌ שגיאה בשליחה לשרת: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ שגיאה בבקשה: {e}")

    cap.release()
    cv2.destroyAllWindows()


def find_available_model():
    """חיפוש מודל זמין במערכת"""
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

    # חיפוש כל קבצי h5
    h5_files = list(Path(".").rglob("*.h5"))
    if h5_files:
        print("📁 קבצי h5 שנמצאו:")
        for i, file in enumerate(h5_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {i + 1}. {file} ({size_mb:.1f} MB)")
            if size_mb > 10:
                print(f"🎯 נבחר: {file}")
                return str(file)

    print("❌ לא נמצא מודל מתאים")
    return None


def create_simple_detector():
    """יצירת גלאי פשוט עם HOG"""
    print("🔧 יוצר גלאי פשוט...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_with_hog(frame, hog_detector):
    """זיהוי אנשים עם HOG עם תצוגה חזותית משופרת"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (rects, weights) = hog_detector.detectMultiScale(
            gray, winStride=(4, 4), padding=(8, 8), scale=1.05, groupThreshold=2
        )

        display_frame = frame.copy()

        # ציור מלבנים סביב אנשים שזוהו
        for i, (x, y, w, h) in enumerate(rects):
            confidence = weights[i] if i < len(weights) else 0.8

            # צבע לפי רמת ביטחון
            if confidence > 0.8:
                color = (0, 255, 0)  # ירוק
            elif confidence > 0.5:
                color = (0, 255, 255)  # צהוב
            else:
                color = (0, 165, 255)  # כתום

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(display_frame, f"Person {confidence:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # הצגת מידע נוסף על המסך
        info_text = [
            f"People Detected: {len(rects)}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"Frame Size: {frame.shape[1]}x{frame.shape[0]}"
        ]

        for i, text in enumerate(info_text):
            y_pos = 30 + (i * 25)
            cv2.putText(display_frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Drone Video - Live Detection', display_frame)
        cv2.waitKey(1)

        # יצירת רשימת אנשים שזוהו
        people = []
        for i, (x, y, w, h) in enumerate(rects):
            confidence = weights[i] if i < len(weights) else 0.8
            if confidence >= CONFIDENCE_THRESHOLD:
                people.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': float(confidence)
                })
        return people

    except Exception as e:
        print(f"❌ שגיאה בזיהוי: {e}")
        cv2.imshow('Drone Video - Live Detection', frame)
        cv2.waitKey(1)
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
            if score < CONFIDENCE_THRESHOLD or int(label) != 0:
                continue
            x1, y1, x2, y2 = map(int, box)
            people.append({"bbox": [x1, y1, x2, y2], "confidence": float(score)})
        return people

    except Exception as e:
        print(f"❌ שגיאת RetinaNet: {e}")
        return []


@dataclass
class Frame:
    number: int
    height: float
    longitude: float
    latitude: float


def load_frames_from_excel(file_path: str) -> List[Frame]:
    """טעינת נתוני פריימים מקובץ Excel עם טיפול בשמות עמודות"""
    try:
        print(f"📖 טוען נתוני מיקום מ-{file_path}...")
        df = pd.read_excel(file_path)

        # תיקון אוטומטי של שגיאות כתיב נפוצות
        column_aliases = {
            'longitute': 'longitude',
            'Longitute': 'longitude',
            'Latitude': 'latitude',
            'Altitude': 'altitude',
        }
        df.rename(columns=column_aliases, inplace=True)

        required_columns = ['timestamp', 'altitude', 'longitude', 'latitude']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ עמודות חסרות: {missing_columns}")
            print(f"📋 עמודות זמינות: {list(df.columns)}")
            return []

        print(f"✅ נמצאו {len(df)} שורות")

        frames = []
        for index, row in df.iterrows():
            try:
                timestamp_str = str(row['timestamp']).replace('s', '')
                timestamp = float(timestamp_str)
                frame = Frame(
                    number=int(timestamp * 30),  # אם FPS = 30
                    height=float(row['altitude']),
                    longitude=float(row['longitude']),
                    latitude=float(row['latitude'])
                )
                frames.append(frame)
            except (ValueError, TypeError) as e:
                print(f"⚠️ שגיאה בשורה {index + 1}: {e}")
                continue

        print(f"✅ טעינה הושלמה: {len(frames)} פריימים תקינים")
        return frames

    except FileNotFoundError:
        print(f"❌ קובץ Excel לא נמצא: {file_path}")
        return []
    except Exception as e:
        print(f"❌ שגיאה בטעינת Excel: {e}")
        return []


def process_video_with_detection(video_path, output_json_path="detection_results.json"):
    """עיבוד וידאו עם זיהוי הולכי רגל - גרסה משופרת עם סנכרון מיקום"""
    print(f"🎬 מעבד וידאו: {video_path}")

    if not Path(video_path).exists():
        print(f"❌ וידאו לא נמצא: {video_path}")
        return False

    # בדיקת חיבור לשרת
    if not check_server_connection():
        print("💡 ניתן להמשיך ללא שרת, אבל הנתונים לא יישמרו")
        return False
    else:
        save_to_server = True
        print("📊 מציג סטטיסטיקות שרת נוכחיות:")
        get_server_stats()

    # טעינת נתוני מיקום מקובץ Excel אם קיים
    location_data = {}
    excel_file = "frames.xlsx"
    if Path(excel_file).exists():
        print(f"📍 טוען נתוני מיקום מ-{excel_file}...")
        try:
            frames = load_frames_from_excel(excel_file)
            for frame in frames:
                location_data[frame.number] = {
                    'height': frame.height,
                    'longitude': frame.longitude,
                    'latitude': frame.latitude
                }
            print(f"✅ נטענו נתוני מיקום עבור {len(location_data)} פריימים")
        except Exception as e:
            print(f"⚠️ שגיאה בטעינת נתוני מיקום: {e}")
            location_data = {}
    else:
        print(f"⚠️ קובץ {excel_file} לא נמצא - מיקומים יהיו 0")

    # טעינת מודל
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

    if model is None:
        print("🔄 עובר לזיהוי HOG...")
        hog_detector = create_simple_detector()
        detection_method = "HOG"
    else:
        hog_detector = None
        detection_method = "RetinaNet"

    # פתיחת וידאו
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ לא ניתן לפתוח את הוידאו")
        return False

    # מידע על הוידאו
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📊 מידע על הוידאו:")
    print(f"   🎯 רזולוציה: {width}x{height}")
    print(f"   ⏱️ FPS: {fps:.2f}")
    print(f"   🎞️ סה״כ פריימים: {total_frames}")
    print(f"   ⏰ משך: {duration:.1f} שניות")
    print(f"   🧠 שיטת זיהוי: {detection_method}")

    # משתנים לעיבוד
    results = []
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    last_progress_time = start_time
    total_people_detected = 0
    failed_uploads = 0
    frames_with_location = 0

    print("🚀 זיהוי החל... לחץ ESC לעצירה")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # בדיקת ESC לעצירה
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⏹️ עצירה על פי בקשת המשתמש")
                break

            # דילוג על פריימים
            if frame_count % SKIP_FRAMES != 0:
                continue

            processed_count += 1

            # זיהוי אנשים
            people = detect_with_retinanet(frame, model) if model else detect_with_hog(frame, hog_detector)
            people_count = len(people)
            total_people_detected += people_count

            # קבלת נתוני מיקום אם קיימים
            location = location_data.get(frame_count, {'height': 0, 'longitude': 0.0, 'latitude': 0.0})

            # בדיקה אם יש מיקום תקין
            has_location = location['latitude'] != 0.0 or location['longitude'] != 0.0
            if has_location:
                frames_with_location += 1

            # יצירת נתוני JSON לפריים
            timestamp_seconds = round(frame_count / fps, 2) if fps > 0 else frame_count
            frame_data = {
                "frame_number": frame_count,
                "timestamp": timestamp_seconds,
                "pedestrian_count": people_count,
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "video_path": video_path,
                "height": location['height'],
                "longitude": location['longitude'],
                "latitude": location['latitude']
            }

            results.append(frame_data)

            # שליחה לשרת
            if save_to_server:
                success = create_video_entry(frame_data)
                if not success:
                    failed_uploads += 1

            # הצגת התקדמות
            current_time = time.time()
            if current_time - last_progress_time >= 2.0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                elapsed = current_time - start_time
                fps_processing = processed_count / elapsed if elapsed > 0 else 0

                location_info = f"📍 {frames_with_location}/{processed_count} עם מיקום" if frames_with_location > 0 else "📍 ללא מיקום"

                print(f"📈 Frame {frame_count:,}/{total_frames:,} ({progress:.1f}%) | "
                      f"זוהו {people_count} אנשים | סה״כ: {total_people_detected} | "
                      f"מהירות: {fps_processing:.1f} FPS | {location_info}")
                last_progress_time = current_time

    except KeyboardInterrupt:
        print("\n⏹️ עצירה על פי בקשת המשתמש (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ שגיאה בעיבוד: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # סיכום עיבוד
    processing_time = time.time() - start_time
    max_people = max((frame['pedestrian_count'] for frame in results), default=0)
    avg_people = total_people_detected / len(results) if results else 0

    print(f"\n🎉 עיבוד הושלם!")
    print(f"⏱️ זמן עיבוד: {processing_time:.1f} שניות")
    print(f"👥 סה״כ זיהויים: {total_people_detected}")
    print(f"🏆 מקסימום בפריים: {max_people}")
    print(f"📈 ממוצע לפריים: {avg_people:.2f}")
    print(f"🎞️ פריימים עובדו: {len(results)}")
    print(
        f"📍 פריימים עם מיקום: {frames_with_location}/{len(results)} ({frames_with_location / len(results) * 100:.1f}%)")

    if save_to_server:
        success_rate = ((len(results) - failed_uploads) / len(results) * 100) if results else 0
        print(f"📤 הועלו לשרת: {len(results) - failed_uploads}/{len(results)} ({success_rate:.1f}%)")

    # שמירת JSON
    try:
        final_result = {
            "video_info": {
                "path": str(video_path),
                "filename": Path(video_path).name,
                "fps": fps,
                "total_frames": total_frames,
                "resolution": {"width": width, "height": height},
                "detection_method": detection_method,
                "duration_seconds": duration
            },
            "summary": {
                "total_pedestrian_detections": total_people_detected,
                "max_pedestrians_in_frame": max_people,
                "average_pedestrians_per_frame": round(avg_people, 2),
                "processing_time_seconds": round(processing_time, 2),
                "frames_with_location": frames_with_location,
                "location_coverage_percent": round(frames_with_location / len(results) * 100, 1) if results else 0
            },
            "frames": results
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        print(f"📁 JSON נשמר ב: {output_json_path}")
    except Exception as e:
        print(f"❌ שגיאה בשמירת JSON: {e}")

    return True


def main():
    """פונקציה ראשית משופרת עם סנכרון מיקום אוטומטי"""
    print("🚁 זיהוי הולכי רגל בסרטוני רחפן - גרסה משופרת")
    print("=" * 60)

    global video_path
    output_json = "pedestrian_detection_results.json"

    # בדיקת קיום וידאו
    if not Path(video_path).exists():
        print(f"❌ וידאו לא נמצא: {video_path}")

        # חיפוש וידאו אלטרנטיבי
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v']
        found_videos = []
        for ext in video_extensions:
            found_videos.extend(list(Path(".").rglob(f"*{ext}")))

        if found_videos:
            print(f"\n📁 נמצאו {len(found_videos)} קבצי וידאו:")
            for i, video in enumerate(found_videos[:10]):
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"  {i + 1}. {video} ({size_mb:.1f} MB)")

            try:
                choice = input(f"\n🔢 בחר מספר וידאו (1-{min(len(found_videos), 10)}) או Enter ליציאה: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= min(len(found_videos), 10):
                    video_path = str(found_videos[int(choice) - 1])
                    print(f"✅ נבחר: {video_path}")
                else:
                    print("👋 יוצא...")
                    return
            except KeyboardInterrupt:
                print("\n👋 יוצא...")
                return
        else:
            print("📂 לא נמצאו קבצי וידאו")
            return

    print(f"\n🎬 עובד עם וידאו: {video_path}")

    # בדיקת קובץ Excel
    excel_file = "frames.xlsx"
    import os
    print(f"📁 מחפש את frames.xlsx בתוך: {os.getcwd()}")

    excel_found = Path(excel_file).exists()

    if excel_found:
        print(f"✅ נמצא קובץ מיקום: {excel_file}")
        print("📍 המיקומים יסונכרנו אוטומטית במהלך העיבוד")
    else:
        print(f"⚠️ קובץ {excel_file} לא נמצא")

        # חיפוש קבצי Excel אלטרנטיביים
        excel_files = list(Path(".").glob("*.xlsx")) + list(Path(".").glob("*.xls"))
        if excel_files:
            print(f"\n📁 נמצאו קבצי Excel אלטרנטיביים:")
            for i, file in enumerate(excel_files):
                print(f"  {i + 1}. {file}")

            try:
                choice = input(f"\nבחר מספר קובץ (1-{len(excel_files)}) או Enter להמשיך ללא מיקום: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(excel_files):
                    excel_file = str(excel_files[int(choice) - 1])
                    print(f"✅ נבחר: {excel_file}")
                    excel_found = True
                else:
                    print("⏭️ ממשיך ללא נתוני מיקום")
                    excel_found = False
            except KeyboardInterrupt:
                print("\n⏭️ ממשיך ללא נתוני מיקום")
                excel_found = False
        else:
            print("📂 לא נמצאו קבצי Excel - ממשיך ללא נתוני מיקום")

    # עיבוד וידאו עם סנכרון מיקום אוטומטי
    print("\n" + "=" * 60)
    print("🔍 מתחיל עיבוד וידאו עם סנכרון מיקום אוטומטי")
    print("=" * 60)

   # success = process_video_with_detection(video_path, output_json)
   # success = create_video_entry(video_path, "client/aerial_pedestrian_detection-master/frames.xlsx")
    success = create_video_entry(video_path, os.path.abspath("frames.xlsx"))

    if not success:
        print("\n❌ עיבוד הוידאו נכשל.")
        return

    print("\n✅ עיבוד הוידאו הושלם בהצלחה!")

    # הצגת הוראות סיום
    print(f"\n" + "=" * 60)
    print("🎊 עיבוד הושלם!")
    print("=" * 60)
    print("📋 מה לעשות עכשיו:")
    print("1. 🗺️ הפעל את המפה הדינמית:")
    print("   python dynamic_map.py")
    print("2. 🌐 גש לכתובת: http://127.0.0.1:8050")
    print("3. 📊 המפה תתעדכן אוטומטית כל 5 שניות")

    if excel_found:
        print("4. 📍 המפה תציג נקודות עם מיקום GPS!")
    else:
        print("4. ⚠️ המפה תציג נקודות ללא מיקום (0,0)")
        print("   💡 להוספת מיקום: הכן קובץ frames.xlsx והפעל שוב")

    # הצגת סטטיסטיקות אחרונות מהשרת
    print(f"\n📊 סטטיסטיקות אחרונות מהשרת:")
    final_stats = get_server_stats()

    if final_stats:
        print("🎯 כל הנתונים מוכנים למפה הדינמית!")
    else:
        print("⚠️ לא ניתן לקבל סטטיסטיקות מהשרת")

    print(f"\n🏆 עיבוד הושלם בהצלחה!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 תוכנית הופסקה על ידי המשתמש")
    except Exception as e:
        print(f"\n\n❌ שגיאה כללית: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n🔚 סיום תוכנית")