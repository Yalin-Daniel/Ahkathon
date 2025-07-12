#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pedestrian detection in drone videos - Fixed Version
Improved connection to the dynamic map - no errors
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

import os
print("Current working directory:", os.getcwd())


# Try importing keras-retinanet
try:
    from keras_retinanet.models import load_model
    from keras_retinanet.utils.visualization import draw_box, label_color
    from keras_retinanet.utils.image import preprocess_image, resize_image

    RETINANET_AVAILABLE = True
    print("keras-retinanet is available")
except ImportError as e:
    print(f"keras-retinanet not available: {e}")
    RETINANET_AVAILABLE = False

# API settings
BASE_URL = "http://localhost:8080"
VIDEO_ENDPOINT = f"{BASE_URL}/insert_video/"
LOCATION_UPDATE_ENDPOINT = f"{BASE_URL}/update_location/"
STATS_ENDPOINT = f"{BASE_URL}/get_video_stats/"

# Video settings
video_path = "video/video1.mp4"
#video_path = r"C:\Users\USER\Desktop\Ahkathon\video\video1.mp4"


# Processing settings
SKIP_FRAMES = 7
CONFIDENCE_THRESHOLD = 0.4


def check_server_connection():
    """Checks connection to the server."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("Server connection successful")
            return True
        else:
            print(f"Server responds but with an error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to the server at {BASE_URL}")
        print("Ensure the server is running on localhost:8080")
        return False
    except requests.exceptions.Timeout:
        print("Server connection timed out")
        return False


def get_server_stats():
    """Retrieves statistics from the server."""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"Server Statistics:")
            print(f"   Total entries: {stats.get('total_entries', 0)}")
            print(f"   Total people detected: {stats.get('total_people_detected', 0)}")
            print(f"   Unique videos: {stats.get('unique_videos', 0)}")
            print(f"   Entries with location: {stats.get('entries_with_location', 0)}")
            return stats
        else:
            print(f"Error getting statistics: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to statistics endpoint: {e}")
        return None


def create_video_entry1(data):
    """Creates a new video entry with improvements."""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(VIDEO_ENDPOINT, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"Frame {data['frame_number']}: Added successfully (ID: {result.get('id')})")
                return True
            else:
                print(f"Frame {data['frame_number']}: Error {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"Retrying {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        except requests.exceptions.ConnectionError:
            print(f"Frame {data['frame_number']}: Connection error")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
        except requests.exceptions.Timeout:
            print(f"Frame {data['frame_number']}: Request timed out")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Frame {data['frame_number']}: Request error - {e}")
            break

    print(f"Frame {data['frame_number']}: Failed after {max_retries} attempts")
    return False

def create_video_entry2(video_path: str, excel_path: str, model=None):
    """
    One function that performs the entire process:
    Load video + location data + pedestrian detection + send with location to server at once.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")

    # Load location from Excel to dictionary
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
        print(f"Loaded {len(location_map)} frames with location")
    except Exception as e:
        print(f"Error loading Excel: {e}")
        location_map = {}

    # Default detector (e.g., HOG)
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

        # Pedestrian detection
        if model is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
            people_count = len(rects)
        else:
            # RetinaNet (if available)
            image = preprocess_image(frame.copy())
            image, scale = resize_image(image)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale
            people_count = sum(1 for score, label in zip(scores[0], labels[0]) if score > 0.4 and label == 0)

        # Time and timestamp
        timestamp_seconds = round(frame_count / fps, 2) if fps > 0 else frame_count
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Location from Excel
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
                print(f"Frame {frame_count}: Sent with {people_count} people and location ({location['latitude']}, {location['longitude']})")
            else:
                print(f"Error sending to server: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"Request error: {e}")

    cap.release()
    cv2.destroyAllWindows()

def create_video_entry(video_path: str, excel_path: str, model=None):
    """
    Load video + location data + pedestrian detection + send with the closest location for each sampled frame.
    """

    def find_closest_location(frame_number, location_map, max_distance=15):
        """Finds the closest GPS location to the frame."""
        if not location_map:
            return {'height': 0, 'longitude': 0.0, 'latitude': 0.0}

        closest_frame = min(location_map.keys(), key=lambda x: abs(x - frame_number))
        if abs(closest_frame - frame_number) > max_distance:
            return {'height': 0, 'longitude': 0.0, 'latitude': 0.0}
        return location_map[closest_frame]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")

    # Load location from Excel file
    try:
        df = pd.read_excel(excel_path)
       # df['timestamp'] = df['timestamp'].astype(str).str.replace('s', '', regex=False).astype(float)
        df['timestamp'] = df['timestamp'].astype(str).str.replace(',', '.', regex=False)
        df['timestamp'] = pd.to_timedelta(df['timestamp']).dt.total_seconds()

        df['frame_number'] = (df['timestamp'] * fps).astype(int)
        location_map = {
            int(row['frame_number']): {
                "height": float(row['altitude']),
                "longitude": float(row['longitude']),
                "latitude": float(row['latitude']),
            }
            for _, row in df.iterrows()
        }
        print(f"Loaded {len(location_map)} frames with location")
    except Exception as e:
        print(f"Error loading Excel: {e}")
        location_map = {}

    # Default detector if RetinaNet model is not available
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
            continue  # Only every N frames

        # Pedestrian detection
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

        # Time
        timestamp_seconds = round(frame_count / fps, 2) if fps > 0 else frame_count
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Get closest location
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
                print(f"Frame {frame_count}: Sent with {people_count} people and location ({location['latitude']}, {location['longitude']})")
            else:
                print(f"Error sending to server: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"Request error: {e}")

    cap.release()
    cv2.destroyAllWindows()


def find_available_model():
    """Searches for an available model in the system."""
    possible_paths = [
        "snapshots/resnet50_csv_08_inference.h5",
        "models/resnet50_csv_08_inference.h5",
        "weights/resnet50_csv_08_inference.h5",
        "../models/resnet50_csv_08_inference.h5",
        "snapshots/resnet50_coco_best_v2.1.0.h5",
    ]

    print("Searching for available models...")
    for path in possible_paths:
        if Path(path).exists():
            print(f"Found model: {path}")
            return path

    # Search for all h5 files
    h5_files = list(Path(".").rglob("*.h5"))
    if h5_files:
        print("Found h5 files:")
        for i, file in enumerate(h5_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {i + 1}. {file} ({size_mb:.1f} MB)")
            if size_mb > 10:
                print(f"Selected: {file}")
                return str(file)

    print("No suitable model found")
    return None


def create_simple_detector():
    """Creates a simple HOG detector."""
    print("Creating a simple detector...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_with_hog(frame, hog_detector):
    """Pedestrian detection with HOG with improved visual display."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (rects, weights) = hog_detector.detectMultiScale(
            gray, winStride=(4, 4), padding=(8, 8), scale=1.05, groupThreshold=2
        )

        display_frame = frame.copy()

        # Draw rectangles around detected people
        for i, (x, y, w, h) in enumerate(rects):
            confidence = weights[i] if i < len(weights) else 0.8

            # Color by confidence level
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(display_frame, f"Person {confidence:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display additional information on the screen
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

        # Create a list of detected people
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
        print(f"Detection error: {e}")
        cv2.imshow('Drone Video - Live Detection', frame)
        cv2.waitKey(1)
        return []


def detect_with_retinanet(frame, model):
    """Detection with RetinaNet (if available)."""
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
        print(f"RetinaNet error: {e}")
        return []


@dataclass
class Frame:
    number: int
    height: float
    longitude: float
    latitude: float


def load_frames_from_excel(file_path: str) -> List[Frame]:
    """Loads frame data from an Excel file with column name handling."""
    try:
        print(f"Loading location data from {file_path}...")
        df = pd.read_excel(file_path)

        # Automatic correction of common spelling mistakes
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
            print(f"Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return []

        print(f"Found {len(df)} rows")

        frames = []
        for index, row in df.iterrows():
            try:
                timestamp_str = str(row['timestamp']).replace('s', '')
                timestamp = float(timestamp_str)
                frame = Frame(
                    number=int(timestamp * 30),  # If FPS = 30
                    height=float(row['altitude']),
                    longitude=float(row['longitude']),
                    latitude=float(row['latitude'])
                )
                frames.append(frame)
            except (ValueError, TypeError) as e:
                print(f"Error in row {index + 1}: {e}")
                continue

        print(f"Loading completed: {len(frames)} valid frames")
        return frames

    except FileNotFoundError:
        print(f"Excel file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return []


def process_video_with_detection(video_path, output_json_path="detection_results.json"):
    """Video processing with pedestrian detection - improved version with location synchronization."""
    print(f"Processing video: {video_path}")

    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return False

    # Check server connection
    if not check_server_connection():
        print("You can continue without a server, but data will not be saved")
        return False
    else:
        save_to_server = True
        print("Displaying current server statistics:")
        get_server_stats()

    # Load location data from Excel file if it exists
    location_data = {}
    excel_file = "frames.xlsx"
    if Path(excel_file).exists():
        print(f"Loading location data from {excel_file}...")
        try:
            frames = load_frames_from_excel(excel_file)
            for frame in frames:
                location_data[frame.number] = {
                    'height': frame.height,
                    'longitude': frame.longitude,
                    'latitude': frame.latitude
                }
            print(f"Loaded location data for {len(location_data)} frames")
        except Exception as e:
            print(f"Error loading location data: {e}")
            location_data = {}
    else:
        print(f"File {excel_file} not found - locations will be 0")

    # Load model
    model = None
    if RETINANET_AVAILABLE:
        model_path = find_available_model()
        if model_path:
            try:
                print(f"Loading RetinaNet model: {model_path}")
                model = load_model(model_path, backbone_name='resnet50')
                print("RetinaNet model loaded successfully")
            except Exception as e:
                print(f"Error loading RetinaNet: {e}")
                model = None

    if model is None:
        print("Switching to HOG detection...")
        hog_detector = create_simple_detector()
        detection_method = "HOG"
    else:
        hog_detector = None
        detection_method = "RetinaNet"

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video")
        return False

    # Video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video Information:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Detection method: {detection_method}")

    # Processing variables
    results = []
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    last_progress_time = start_time
    total_people_detected = 0
    failed_uploads = 0
    frames_with_location = 0

    print("Detection started... Press ESC to stop")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Check ESC to stop
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\nStopping at user request")
                break

            # Skip frames
            if frame_count % SKIP_FRAMES != 0:
                continue

            processed_count += 1

            # Pedestrian detection
            people = detect_with_retinanet(frame, model) if model else detect_with_hog(frame, hog_detector)
            people_count = len(people)
            total_people_detected += people_count

            # Get location data if available
            location = location_data.get(frame_count, {'height': 0, 'longitude': 0.0, 'latitude': 0.0})

            # Check if there is valid location
            has_location = location['latitude'] != 0.0 or location['longitude'] != 0.0
            if has_location:
                frames_with_location += 1

            # Create JSON data for the frame
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

            # Send to server
            if save_to_server:
                success = create_video_entry(frame_data)
                if not success:
                    failed_uploads += 1

            # Display progress
            current_time = time.time()
            if current_time - last_progress_time >= 2.0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                elapsed = current_time - start_time
                fps_processing = processed_count / elapsed if elapsed > 0 else 0

                location_info = f"{frames_with_location}/{processed_count} with location" if frames_with_location > 0 else "No location"

                print(f"Frame {frame_count:,}/{total_frames:,} ({progress:.1f}%) | "
                      f"Detected {people_count} people | Total: {total_people_detected} | "
                      f"Speed: {fps_processing:.1f} FPS | {location_info}")
                last_progress_time = current_time

    except KeyboardInterrupt:
        print("\nStopping at user request (Ctrl+C)")
    except Exception as e:
        print(f"\nError processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Processing summary
    processing_time = time.time() - start_time
    max_people = max((frame['pedestrian_count'] for frame in results), default=0)
    avg_people = total_people_detected / len(results) if results else 0

    print(f"\nProcessing completed!")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Total detections: {total_people_detected}")
    print(f"Maximum in frame: {max_people}")
    print(f"Average per frame: {avg_people:.2f}")
    print(f"Processed frames: {len(results)}")
    print(
        f"Frames with location: {frames_with_location}/{len(results)} ({frames_with_location / len(results) * 100:.1f}%)")

    if save_to_server:
        success_rate = ((len(results) - failed_uploads) / len(results) * 100) if results else 0
        print(f"Uploaded to server: {len(results) - failed_uploads}/{len(results)} ({success_rate:.1f}%)")

    # Save JSON
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
        print(f"JSON saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    return True


def main():
    """Main function improved with automatic location synchronization."""
    print("Drone Pedestrian Detection - Improved Version")
    print("=" * 60)

    global video_path
    output_json = "pedestrian_detection_results.json"

    # Check video existence
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")

        # Search for alternative videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v']
        found_videos = []
        for ext in video_extensions:
            found_videos.extend(list(Path(".").rglob(f"*{ext}")))

        if found_videos:
            print(f"\nFound {len(found_videos)} video files:")
            for i, video in enumerate(found_videos[:10]):
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"  {i + 1}. {video} ({size_mb:.1f} MB)")

            try:
                choice = input(f"\nSelect video number (1-{min(len(found_videos), 10)}) or Enter to exit: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= min(len(found_videos), 10):
                    video_path = str(found_videos[int(choice) - 1])
                    print(f"Selected: {video_path}")
                else:
                    print("Exiting...")
                    return
            except KeyboardInterrupt:
                print("\nExiting...")
                return
        else:
            print("No video files found")
            return

    print(f"\nWorking with video: {video_path}")

    # Check Excel file
    excel_file = "frames.xlsx"
    import os
    print(f"Looking for frames.xlsx in: {os.getcwd()}")

    excel_found = Path(excel_file).exists()

    if excel_found:
        print(f"Location file found: {excel_file}")
        print("Locations will be automatically synchronized during processing")
    else:
        print(f"File {excel_file} not found")

        # Search for alternative Excel files
        excel_files = list(Path(".").glob("*.xlsx")) + list(Path(".").glob("*.xls"))
        if excel_files:
            print(f"\nFound alternative Excel files:")
            for i, file in enumerate(excel_files):
                print(f"  {i + 1}. {file}")

            try:
                choice = input(f"\nSelect file number (1-{len(excel_files)}) or Enter to continue without location: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(excel_files):
                    excel_file = str(excel_files[int(choice) - 1])
                    print(f"Selected: {excel_file}")
                    excel_found = True
                else:
                    print("Continuing without location data")
                    excel_found = False
            except KeyboardInterrupt:
                print("\nContinuing without location data")
                excel_found = False
        else:
            print("No Excel files found - continuing without location data")

    # Video processing with automatic location synchronization
    print("\n" + "=" * 60)
    print("Starting video processing with automatic location synchronization")
    print("=" * 60)

   # success = process_video_with_detection(video_path, output_json)
   # success = create_video_entry(video_path, "client/aerial_pedestrian_detection-master/frames.xlsx")
    success = create_video_entry(video_path, os.path.abspath("frames.xlsx"))

    if not success:
        print("\nVideo processing failed.")
        return

    print("\nVideo processing completed successfully!")

    # Display final instructions
    print(f"\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)
    print("What to do next:")
    print("1. Run the dynamic map:")
    print("   python dynamic_map.py")
    print("2. Go to: http://127.0.0.1:8050")
    print("3. The map will update automatically every 5 seconds")

    if excel_found:
        print("4. The map will display points with GPS location!")
    else:
        print("4. The map will display points without location (0,0)")
        print("   To add location: prepare a frames.xlsx file and run again")

    # Display latest server statistics
    print(f"\nLatest server statistics:")
    final_stats = get_server_stats()

    if final_stats:
        print("All data is ready for the dynamic map!")
    else:
        print("Could not get statistics from the server")

    print(f"\nProcessing completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user")
    except Exception as e:
        print(f"\n\nGeneral error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nEnd of program")