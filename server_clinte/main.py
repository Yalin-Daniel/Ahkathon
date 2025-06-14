# main.py - השרת המעודכן
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import List, Optional
import os
import json
import shutil
from pathlib import Path
VIDEO_DIR = Path("../video")  # ⬅ שים לב! נתיב יחסי מהמיקום של main.py

app = FastAPI(title="Drone Data Analysis System")

# Database configuration
DATABASE_URL = 'postgresql://neondb_owner:npg_0lQmgDuUfM8S@ep-withered-queen-a29pkpay-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require'

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# יצירת תיקיות לסרטונים ולתוצאות
VIDEO_DIR = Path("uploaded_videos")
RESULTS_DIR = Path("results_json")
VIDEO_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# Database Models (מעודכן)
class VideoEntryDB(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    video_filename = Column(String, index=True)
    capture_time = Column(DateTime)
    drone_latitude = Column(Float)
    drone_longitude = Column(Float)
    altitude_m = Column(Float)
    fov_horizontal = Column(Float, default=90.0)
    fov_vertical = Column(Float, default=72.0)

    # קואורדינטות שדה הראייה (4 פינות)
    footprint_ne_lat = Column(Float)
    footprint_ne_lon = Column(Float)
    footprint_se_lat = Column(Float)
    footprint_se_lon = Column(Float)
    footprint_sw_lat = Column(Float)
    footprint_sw_lon = Column(Float)
    footprint_nw_lat = Column(Float)
    footprint_nw_lon = Column(Float)

    # תוצאות זיהוי
    people_count = Column(Integer)
    people_probability = Column(Float)
    detection_results = Column(Text)  # JSON של כל הזיהויים
    grid_id = Column(String)
    processing_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)


# Pydantic Models (Request/Response schemas)
class VideoProcessRequest(BaseModel):
    video_filename: str
    capture_time: datetime
    drone_lat: float
    drone_lon: float
    altitude_m: float
    fov_horizontal: Optional[float] = 90.0
    fov_vertical: Optional[float] = 72.0
    grid_id: str


class VideoEntry(BaseModel):
    """למי שרוצה להזין ידנית"""
    capture_time: datetime
    lat: float
    lon: float
    people_count: int
    people_probability: float
    grid_id: str


class UserCreate(BaseModel):
    name: str
    email: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========== ENDPOINTS ===========

# 1. העלאת סרטון
@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """העלאת סרטון לעיבוד"""
    try:
        # שמור קובץ
        file_path = VIDEO_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "filename": file.filename,
            "message": "קובץ הועלה בהצלחה. עכשיו שלח בקשת עיבוד."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"שגיאה בהעלאת קובץ: {e}")


# 2. עיבוד סרטון מלא (החדש!)
@app.post("/process_video/")
def process_video(request: VideoProcessRequest, db: Session = Depends(get_db)):
    """עיבוד סרטון מלא - זיהוי + שמירה בDB"""

    # ייבוא כל הפונקציות הנדרשות
    from video_processor import process_video_with_detection
    from footprint_calculator import calculate_footprint

    video_path = VIDEO_DIR / request.video_filename

    # בדוק שהקובץ קיים
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"סרטון לא נמצא: {request.video_filename}")

    try:
        # 1. חשב שדה ראייה
        footprint = calculate_footprint(
            request.drone_lat,
            request.drone_lon,
            request.altitude_m,
            request.fov_horizontal,
            request.fov_vertical
        )

        # 2. הרץ זיהוי על הסרטון
        output_json = RESULTS_DIR / f"{request.video_filename}_results.json"
        success = process_video_with_detection(str(video_path), str(output_json))

        if not success:
            raise HTTPException(status_code=500, detail="שגיאה בעיבוד הסרטון")

        # 3. קרא תוצאות זיהוי
        with open(output_json, 'r', encoding='utf-8') as f:
            detection_data = json.load(f)

        # 4. חלץ סטטיסטיקות
        people_count = detection_data['summary']['max_pedestrians_in_frame']
        avg_confidence = 0.85  # ממוצע או חישוב מדויק יותר

        # 5. שמור בבסיס נתונים
        db_entry = VideoEntryDB(
            video_filename=request.video_filename,
            capture_time=request.capture_time,
            drone_latitude=request.drone_lat,
            drone_longitude=request.drone_lon,
            altitude_m=request.altitude_m,
            fov_horizontal=request.fov_horizontal,
            fov_vertical=request.fov_vertical,

            # שדה ראייה
            footprint_ne_lat=footprint['NE'][0],
            footprint_ne_lon=footprint['NE'][1],
            footprint_se_lat=footprint['SE'][0],
            footprint_se_lon=footprint['SE'][1],
            footprint_sw_lat=footprint['SW'][0],
            footprint_sw_lon=footprint['SW'][1],
            footprint_nw_lat=footprint['NW'][0],
            footprint_nw_lon=footprint['NW'][1],

            # תוצאות זיהוי
            people_count=people_count,
            people_probability=avg_confidence,
            detection_results=json.dumps(detection_data),
            grid_id=request.grid_id,
            processing_status="completed"
        )

        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        return {
            "status": "success",
            "video_id": db_entry.id,
            "people_detected": people_count,
            "footprint": footprint,
            "message": "סרטון עובד בהצלחה ונשמר בבסיס הנתונים"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"שגיאה בעיבוד: {str(e)}")


# 3. קבלת כל הסרטונים
@app.get("/videos/")
def get_all_videos(db: Session = Depends(get_db)):
    """קבלת רשימת כל הסרטונים שעובדו"""
    videos = db.query(VideoEntryDB).all()

    result = []
    for video in videos:
        result.append({
            "id": video.id,
            "filename": video.video_filename,
            "capture_time": video.capture_time,
            "drone_position": {
                "lat": video.drone_latitude,
                "lon": video.drone_longitude,
                "altitude": video.altitude_m
            },
            "people_count": video.people_count,
            "processing_status": video.processing_status,
            "grid_id": video.grid_id,
            "created_at": video.created_at
        })

    return result


# 4. קבלת סרטון ספציפי עם פרטים מלאים
@app.get("/videos/{video_id}")
def get_video_details(video_id: int, db: Session = Depends(get_db)):
    """קבלת פרטים מלאים על סרטון כולל תוצאות זיהוי"""
    video = db.query(VideoEntryDB).filter(VideoEntryDB.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="סרטון לא נמצא")

    # המרת JSON של תוצאות זיהוי
    detection_results = None
    if video.detection_results:
        detection_results = json.loads(video.detection_results)

    return {
        "id": video.id,
        "filename": video.video_filename,
        "capture_time": video.capture_time,
        "drone_position": {
            "lat": video.drone_latitude,
            "lon": video.drone_longitude,
            "altitude": video.altitude_m
        },
        "camera_settings": {
            "fov_horizontal": video.fov_horizontal,
            "fov_vertical": video.fov_vertical
        },
        "footprint": {
            "NE": [video.footprint_ne_lat, video.footprint_ne_lon],
            "SE": [video.footprint_se_lat, video.footprint_se_lon],
            "SW": [video.footprint_sw_lat, video.footprint_sw_lon],
            "NW": [video.footprint_nw_lat, video.footprint_nw_lon]
        },
        "detection_summary": {
            "people_count": video.people_count,
            "confidence": video.people_probability
        },
        "detection_results": detection_results,
        "grid_id": video.grid_id,
        "processing_status": video.processing_status,
        "created_at": video.created_at
    }


# 5. נקודות קצה ישנות - תואמות לאחור
@app.post("/insert_video/")
def insert_video(entry: VideoEntry, db: Session = Depends(get_db)):
    """הזנה ידנית (תואמות לאחור)"""
    from footprint_calculator import calculate_footprint

    # חישוב footprint עם ברירות מחדל
    footprint = calculate_footprint(entry.lat, entry.lon, 100)  # גובה ברירת מחדל

    db_entry = VideoEntryDB(
        video_filename="manual_entry",
        capture_time=entry.capture_time,
        drone_latitude=entry.lat,
        drone_longitude=entry.lon,
        altitude_m=100,  # ברירת מחדל
        footprint_ne_lat=footprint['NE'][0],
        footprint_ne_lon=footprint['NE'][1],
        footprint_se_lat=footprint['SE'][0],
        footprint_se_lon=footprint['SE'][1],
        footprint_sw_lat=footprint['SW'][0],
        footprint_sw_lon=footprint['SW'][1],
        footprint_nw_lat=footprint['NW'][0],
        footprint_nw_lon=footprint['NW'][1],
        people_count=entry.people_count,
        people_probability=entry.people_probability,
        grid_id=entry.grid_id,
        processing_status="manual"
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return {"status": "success", "id": db_entry.id}


# User endpoints (ללא שינוי)
@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
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
    users = db.query(UserDB).all()
    return users


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# בדיקת בריאות
@app.get("/")
def root():
    return {"message": "Drone Analysis System is running!", "status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)