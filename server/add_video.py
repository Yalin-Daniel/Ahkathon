import requests
from datetime import datetime
import json

# API endpoint URL
BASE_URL = "http://localhost:8080"
VIDEO_ENDPOINT = f"{BASE_URL}/insert_video/"


def create_video_entry(capture_time, lat, lon, people_count, people_probability,grid_id):
    """
    Create a new video entry by sending a POST request to the API
    """
    # Prepare the data payload
    data = {
        "capture_time": capture_time.isoformat(),
        "lat": lat,
        "lon": lon,
        "people_count": people_count,
        "people_probability": people_probability,
        "grid_id": grid_id
    }

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


def main():
    """
    Main function to create sample video entries
    """
    print("Creating video entries...")

    # Example 1: Current time entry
    create_video_entry(
        capture_time=datetime.now(),
        lat=32.0853,  # Tel Aviv latitude
        lon=34.7818,  # Tel Aviv longitude
        people_count=5,
        people_probability=0.85,
        grid_id='gg5g'
    )

    # Example 2: Specific time entry
    create_video_entry(
        capture_time=datetime(2024, 12, 15, 14, 30, 0),
        lat=31.7683,  # Jerusalem latitude
        lon=35.2137,  # Jerusalem longitude
        people_count=3,
        people_probability=0.92,
        grid_id='123'
    )

    # Example 3: Another entry
    create_video_entry(
        capture_time=datetime(2024, 12, 15, 16, 45, 30),
        lat=32.8156,  # Haifa latitude
        lon=34.9892,  # Haifa longitude
        people_count=7,
        people_probability=0.78,
        grid_id='aaa'
    )


if __name__ == "__main__":
    main()