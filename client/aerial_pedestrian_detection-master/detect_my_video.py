#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pedestrian detection in drone videos - Corrected Version
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

from PIL.TiffImagePlugin import DATE_TIME

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

BASE_URL = "http://localhost:8080"
VIDEO_ENDPOINT = f"{BASE_URL}/insert_video/"

def create_video_entry(data):
    """
    Create a new video entry by sending a POST request to the API
    """
    # Prepare the data payload
    # data = {
    #     "capture_time": capture_time.isoformat(),
    #     "lat": lat,
    #     "lon": lon,
    #     "people_count": people_count,
    #     "people_probability": people_probability,
    #     "grid_id": grid_id
    # }

    try:
        # Send POST request
        response = requests.post(VIDEO_ENDPOINT, json=data)

        # Check if request was successful
        if response.status_code == 200:
            print("✅ Video entry created successfully!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Error creating video entry: {response.status_code}")
            print(f"Error details: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error: Make sure the FastAPI server is running on {BASE_URL} ")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")


def find_available_model():
    """
    Search for an available model in the system
    """
    # Possible paths for models
    possible_paths = [
        "snapshots/resnet50_csv_08_inference.h5",
        "models/resnet50_csv_08_inference.h5",
        "weights/resnet50_csv_08_inference.h5",
        "../models/resnet50_csv_08_inference.h5",
        "snapshots/resnet50_coco_best_v2.1.0.h5",  # General COCO model
    ]

    print("🔍 Searching for available models...")

    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Found model: {path}")
            return path

    # Search for all .h5 files
    h5_files = list(Path(".").rglob("*.h5"))
    if h5_files:
        print("📁 .h5 files found:")
        for i, file in enumerate(h5_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {i + 1}. {file} ({size_mb:.1f} MB)")

        # Take the first one that's big enough
        for file in h5_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # Real model is probably larger than 10MB
                print(f"🎯 Selected: {file}")
                return str(file)

    print("❌ No suitable model found")
    return None


def create_simple_detector():
    """
    Create simple detection that works without complex model
    """
    print("🔧 Creating simple detector...")

    # HOG + SVM for people detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    return hog

def detect_with_hog(frame, hog_detector):
    """
    People detection with HOG with visual display
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection with HOG
        (rects, weights) = hog_detector.detectMultiScale(
            gray,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            groupThreshold=2
        )

        # Create copy for display
        display_frame = frame.copy()

        # Draw rectangles around detected people
        for (x, y, w, h) in rects:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display_frame, "Person", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display number of people on screen
        cv2.putText(display_frame, f"People Detected: {len(rects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display frame
        cv2.imshow('Drone Video - Live Detection', display_frame)
        cv2.waitKey(1)

        # Create list of detected people
        people = []
        for i, (x, y, w, h) in enumerate(rects):
            confidence = 0.8  # Fixed value instead of weights
            people.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': confidence
            })

        return people

    except Exception as e:
        print(f"Detection error: {e}")
        # Display frame even in case of error
        cv2.imshow('Drone Video - Live Detection', frame)
        cv2.waitKey(1)
        return []

def detect_with_retinanet(frame, model):
    """
    Detection with RetinaNet (if available)
    """
    try:
        # Image processing
        image = preprocess_image(frame.copy())
        image, scale = resize_image(image)

        # Prediction
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        people = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.4:
                continue
            if int(label) != 0:  # 0 = person in COCO
                continue

            x1, y1, x2, y2 = map(int, box)
            people.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score)
            })

        return people

    except Exception as e:
        print(f"❌ RetinaNet error: {e}")
        return []


def process_video_with_detection(video_path, output_json_path="detection_results.json"):
    """
    Process video with pedestrian detection
    """
    print(f"🎬 Processing video: {video_path}")

    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return False

    # Try to load RetinaNet model
    model = None
    if RETINANET_AVAILABLE:
        model_path = find_available_model()
        if model_path:
            try:
                print(f"📥 Loading RetinaNet model: {model_path}")
                model = load_model(model_path, backbone_name='resnet50')
                print("✅ RetinaNet model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading RetinaNet: {e}")
                model = None

    # If RetinaNet didn't work, use HOG
    if model is None:
        print("🔄 Switching to HOG detection...")
        hog_detector = create_simple_detector()
        detection_method = "HOG"
    else:
        hog_detector = None
        detection_method = "RetinaNet"

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return False

    # Video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📊 Video information:")
    print(f"   🎯 Resolution: {width}x{height}")
    print(f"   ⏱️  FPS: {fps}")
    print(f"   🎞️ Total frames: {total_frames}")
    print(f"   🧠 Detection method: {detection_method}")

    # Results directory
    results_dir = Path("results_json")
    results_dir.mkdir(exist_ok=True)

    results = []
    frame_count = 0
    SKIP_FRAMES = 7  # Process every 3rd frame
    start_time = time.time()

    print("🚀 Detection started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % SKIP_FRAMES != 0:
            continue

        # People detection
        if model is not None:
            people = detect_with_retinanet(frame, model)
        else:
            people = detect_with_hog(frame, hog_detector)

        # Create JSON data for frame
        frame_data = {
            "frame_number": frame_count,
            "timestamp": int(round(frame_count / fps, 2)) if fps > 0 else frame_count,
            "pedestrian_count": len(people),
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_path": video_path
        }
#             "detections": [
#     {
#         "detection_id": i,
#         "confidence": person["confidence"],  # ← Fixed!
#         # "bounding_box": {
#         #     "x": person["bbox"][0],
#         #     "y": person["bbox"][1],
#         #     "width": person["bbox"][2] - person["bbox"][0],
#         #     "height": person["bbox"][3] - person["bbox"][1]
#         # }
#     }
#     for i, person in enumerate(people)
# ],
#             "metadata": {
#                 "frame_width": width,
#                 "frame_height": height,
#                 "detection_method": detection_method,
#                 "processing_successful": True
#             }
#         }

        results.append(frame_data)

        # Save individual frame JSON
        frame_json_path = results_dir / f"frame_{frame_count:05d}.json"
        # todo: replace with post request
        # with open(frame_json_path, "w", encoding='utf-8') as f:
        #     json.dump(frame_data, f, indent=2, ensure_ascii=False)
        try:
            # Send POST request
            response = requests.post(VIDEO_ENDPOINT, json=frame_data)
            print("ℹ️ frame data: ", frame_data)
            # Check if request was successful
            if response.status_code == 200:
                print("✅ Video entry created successfully!")
                print(f"Response: {response.json()}")
            else:
                print(f"❌ Error creating video entry: {response.status_code}")
                print(f"Error details: {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error: Make sure the FastAPI server is running on {BASE_URL} ")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")

        # Print progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            elapsed = time.time() - start_time
            print(f"📈 Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Detected {len(people)} people | "
                  f"Time: {elapsed:.1f}s")

    cap.release()

    # Save complete JSON
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

    # Save complete result
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Processing completed!")
    print(f"📁 Complete JSON saved at: {output_json_path}")
    print(f"📁 Frame JSONs in: {results_dir}")
    print(f"📊 Statistics:")
    print(f"   👥 Total detections: {total_detections}")
    print(f"   🏆 Maximum in frame: {max_people}")
    print(f"   📈 Average per frame: {avg_people:.2f}")
    print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")

    return True

@dataclass
class Frame:
    number: int
    height: float
    longitude: float
    latitude: float

def load_frames_from_excel(file_path: str):
    df = pd.read_excel(file_path)
    frames = [
        Frame(
            number=row['timestamp'],
            height=row['altitude'],
            longitude=row['longitude'],
            latitude=row['latitude']
        )
        for _, row in df.iterrows()
    ]
    return frames

# Usage
if __name__ == "__main__":
    print("🚁 Pedestrian Detection in Drone Videos")
    print("=" * 50)

    # Set paths
    video_path = "video/video1.mp4"  # Update the path
    output_json = "pedestrian_detection_results.json"

    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("💡 Update the path in video_path")

        # Search for videos in area
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        found_videos = []

        for ext in video_extensions:
            found_videos.extend(list(Path(".").rglob(f"*{ext}")))
            found_videos.extend(list(Path("..").rglob(f"*{ext}")))
        #TODO CHOOSE A LIST OF VIDEOS
        if found_videos:
            print("\n📁 Videos found:")
            for i, video in enumerate(found_videos[:5]):  # Show first 5
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"  {i + 1}. {video} ({size_mb:.1f} MB)")

            # Give option to choose
            try:
                choice = input("\n🔢 Choose video number (or Enter to exit): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(found_videos):
                    video_path = str(found_videos[int(choice) - 1])
                    print(f"✅ Selected: {video_path}")
                else:
                    exit()
            except:
                exit()

    # Run detection
    success = process_video_with_detection(video_path, output_json)

    if success:
        print("\n🎊 System is working! Now you can connect to your system")
    else:
        print("\n❌ There's a problem. Try with another video or check errors")
    frams = load_frames_from_excel("frames.xlsx")

