from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, desc
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import List, Optional
import os

# Required imports
from sqlalchemy import func
from datetime import timedelta
import uvicorn

app = FastAPI(title="Drone Video Analysis API", version="1.0.0")

# Database configuration
DATABASE_URL = 'postgresql://neondb_owner:npg_0lQmgDuUfM8S@ep-withered-queen-a29pkpay-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require'

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class VideoEntryDB(Base):
    """SQLAlchemy model for video entries."""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    frame_number = Column(Integer, index=True)
    timestamp = Column(Float, index=True)
    pedestrian_count = Column(Integer, index=True)
    time = Column(String)
    video_path = Column(String)
    height = Column(Float)
    longitude = Column(Float)
    latitude = Column(Float)


class UserDB(Base):
    """SQLAlchemy model for users."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create database tables
Base.metadata.create_all(bind=engine)


# Pydantic Models (Request/Response schemas)
class VideoEntry(BaseModel):
    """Pydantic model for creating a new video entry."""
    frame_number: int
    timestamp: float
    pedestrian_count: int
    time: str
    video_path: str
    height: float
    longitude: float
    latitude: float


class VideoEntryResponse(BaseModel):
    """Pydantic model for returning video entry data."""
    id: int
    frame_number: int
    timestamp: float
    pedestrian_count: int
    time: str
    video_path: str
    height: float
    longitude: float
    latitude: float

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """Pydantic model for creating a new user."""
    name: str
    email: str


class UserResponse(BaseModel):
    """Pydantic model for returning user data."""
    id: int
    name: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class LocationUpdate(BaseModel):
    """Pydantic model for updating location data of a video entry."""
    frame_number: int
    video_path: str
    height: float
    longitude: float
    latitude: float


class VideoStats(BaseModel):
    """Pydantic model for returning video statistics."""
    total_entries: int
    total_people_detected: int
    unique_videos: int
    entries_with_location: int
    latest_entry_time: Optional[str]
    average_people_per_frame: float


# Dependency to get database session
def get_db():
    """Dependency to provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Video endpoints
@app.post("/insert_video/", response_model=dict)
def insert_video(entry: VideoEntry, db: Session = Depends(get_db)):
    """
    Adds a new video entry to the database.
    """
    print(f"Request to add frame {entry.frame_number}")

    db_entry = VideoEntryDB(
        frame_number=entry.frame_number,
        timestamp=entry.timestamp,
        pedestrian_count=entry.pedestrian_count,
        time=entry.time,
        video_path=entry.video_path,
        height=float(entry.height),
        longitude=entry.longitude,
        latitude=entry.latitude
    )

    try:
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        print(f"Added frame {entry.frame_number} to the database (ID: {db_entry.id})")
        return {"status": "success", "id": db_entry.id}

    except Exception as e:
        db.rollback()
        print(f"Error adding frame {entry.frame_number}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to insert video entry: {str(e)}")


@app.get("/get_all_videos/", response_model=List[VideoEntryResponse])
def get_all_videos(
        limit: Optional[int] = 1000,
        only_with_location: bool = False,
        db: Session = Depends(get_db)
):
    """
    Retrieves all video entries (with optional filtering).
    """
    try:
        query = db.query(VideoEntryDB)

        if only_with_location:
            # Filter for entries with valid location data
            query = query.filter(
                VideoEntryDB.latitude != 0,
                VideoEntryDB.longitude != 0
            )

        # Order by time (most recent first)
        query = query.order_by(desc(VideoEntryDB.id))

        # Limit the number of results
        if limit:
            query = query.limit(limit)

        videos = query.all()
        print(f"Returning {len(videos)} video entries")
        return videos

    except Exception as e:
        print(f"Error fetching video data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch videos: {str(e)}")


@app.get("/get_videos_by_path/{video_path:path}", response_model=List[VideoEntryResponse])
def get_videos_by_path(video_path: str, db: Session = Depends(get_db)):
    """
    Retrieves video entries by a specific video path.
    """
    try:
        videos = db.query(VideoEntryDB).filter(
            VideoEntryDB.video_path == video_path
        ).order_by(VideoEntryDB.frame_number).all()

        print(f"Returning {len(videos)} entries for video: {video_path}")
        return videos

    except Exception as e:
        print(f"Error fetching data for video {video_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch videos: {str(e)}")


@app.get("/get_recent_videos/", response_model=List[VideoEntryResponse])
def get_recent_videos(minutes: int = 60, db: Session = Depends(get_db)):
    """
    Retrieves video entries from the last few hours.
    """
    try:
        # Calculate the time threshold
        time_threshold = datetime.now() - timedelta(minutes=minutes)

        videos = db.query(VideoEntryDB).filter(
            VideoEntryDB.time >= time_threshold.strftime('%Y-%m-%d %H:%M:%S')
        ).order_by(desc(VideoEntryDB.id)).all()

        print(f"Returning {len(videos)} entries from the last {minutes} minutes")
        return videos

    except Exception as e:
        print(f"Error fetching recent entries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent videos: {str(e)}")


@app.get("/get_video_stats/", response_model=VideoStats)
def get_video_stats(db: Session = Depends(get_db)):
    """
    Retrieves general statistics about the video data.
    """
    try:
        # Total number of entries
        total_entries = db.query(VideoEntryDB).count()

        # Total number of people detected
        total_people = db.query(func.sum(VideoEntryDB.pedestrian_count)).scalar() or 0

        # Number of unique videos
        unique_videos = db.query(VideoEntryDB.video_path).distinct().count()

        # Entries with valid location data
        entries_with_location = db.query(VideoEntryDB).filter(
            VideoEntryDB.latitude != 0,
            VideoEntryDB.longitude != 0
        ).count()

        # Latest entry time
        latest_entry = db.query(VideoEntryDB).order_by(desc(VideoEntryDB.id)).first()
        latest_time = latest_entry.time if latest_entry else None

        # Average people per frame
        avg_people = total_people / total_entries if total_entries > 0 else 0

        stats = VideoStats(
            total_entries=total_entries,
            total_people_detected=total_people,
            unique_videos=unique_videos,
            entries_with_location=entries_with_location,
            latest_entry_time=latest_time,
            average_people_per_frame=round(avg_people, 2)
        )

        print(f"Statistics: {total_entries} entries, {total_people} people")
        return stats

    except Exception as e:
        print(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate stats: {str(e)}")


@app.put("/update_location/", response_model=dict)
def update_video_location(location_data: LocationUpdate, db: Session = Depends(get_db)):
    """
    Updates only the location data for an existing video entry in the database.
    """
    print(f"Updating location for frame {location_data.frame_number}")

    # Find the existing entry
    existing_entry = db.query(VideoEntryDB).filter(
        VideoEntryDB.frame_number == location_data.frame_number,
        VideoEntryDB.video_path == location_data.video_path
    ).first()

    if not existing_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Video entry not found for frame {location_data.frame_number} in video {location_data.video_path}"
        )

    # Update location fields
    existing_entry.height = location_data.height
    existing_entry.longitude = location_data.longitude
    existing_entry.latitude = location_data.latitude

    try:
        db.commit()
        db.refresh(existing_entry)

        print(f"Location updated for frame {location_data.frame_number}")
        return {
            "status": "success",
            "updated_id": existing_entry.id,
            "frame_number": existing_entry.frame_number,
            "message": "Location data updated successfully"
        }

    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update database: {str(e)}")


@app.delete("/delete_video/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    """
    Deletes a specific video entry.
    """
    try:
        video = db.query(VideoEntryDB).filter(VideoEntryDB.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video entry not found")

        db.delete(video)
        db.commit()

        print(f"Deleted entry {video_id}")
        return {"status": "success", "message": f"Video entry {video_id} deleted"}

    except Exception as e:
        db.rollback()
        print(f"Error deleting entry {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")


# User endpoints
@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Creates a new user.
    """
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    db_user = UserDB(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/users/", response_model=List[UserResponse])
def get_all_users(db: Session = Depends(get_db)):
    """
    Retrieves all users.
    """
    users = db.query(UserDB).all()
    return users


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a specific user by ID.
    """
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/")
def root():
    """
    Main endpoint providing information about the API.
    """
    return {
        "message": "Drone Video Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /insert_video/": "Add a video entry",
            "GET /get_all_videos/": "Retrieve all entries",
            "GET /get_videos_by_path/{path}": "Retrieve entries by video path",
            "GET /get_recent_videos/": "Retrieve recent entries",
            "GET /get_video_stats/": "General statistics",
            "PUT /update_location/": "Update location data",
            "DELETE /delete_video/{id}": "Delete a specific entry"
        }
    }

@app.delete("/delete_all_videos/")
def delete_all_videos(db: Session = Depends(get_db)):
    """
    Deletes all video entries from the database.
    """
    try:
        deleted = db.query(VideoEntryDB).delete()
        db.commit()
        print(f"Deleted {deleted} entries from the 'videos' table")
        return {"status": "success", "message": f"Deleted {deleted} video entries"}
    except Exception as e:
        db.rollback()
        print(f"Error deleting all entries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete all videos: {str(e)}")


# Main function to run the server
if __name__ == "__main__":
    print("Starting API server...")
    print("Server will run on: http://localhost:8080")
    print("Application documentation: http://localhost:8080/docs")

    uvicorn.run("server.api_server:app", host="127.0.0.1", port=8080, reload=True)