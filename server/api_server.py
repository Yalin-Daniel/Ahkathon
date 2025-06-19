from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, desc
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import List, Optional
import os

app = FastAPI(title="Drone Video Analysis API", version="1.0.0")

# Database configuration
DATABASE_URL = 'postgresql://neondb_owner:npg_0lQmgDuUfM8S@ep-withered-queen-a29pkpay-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require'

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class VideoEntryDB(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    frame_number = Column(Integer, index=True)
    timestamp = Column(Float, index=True)
    pedestrian_count = Column(Integer, index=True)
    time = Column(String)
    video_path = Column(String)
    height = Column(Float)  # שונה ל-Float במקום Integer
    longitude = Column(Float)
    latitude = Column(Float)


class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)


# Pydantic Models (Request/Response schemas)
class VideoEntry(BaseModel):
    frame_number: int
    timestamp: float
    pedestrian_count: int
    time: str
    video_path: str
    height: int
    longitude: float
    latitude: float


class VideoEntryResponse(BaseModel):
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
    name: str
    email: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class LocationUpdate(BaseModel):
    frame_number: int
    video_path: str
    height: float
    longitude: float
    latitude: float


class VideoStats(BaseModel):
    total_entries: int
    total_people_detected: int
    unique_videos: int
    entries_with_location: int
    latest_entry_time: Optional[str]
    average_people_per_frame: float


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Video endpoints
@app.post("/insert_video/", response_model=dict)
def insert_video(entry: VideoEntry, db: Session = Depends(get_db)):
    """
    הוספת רשומת וידאו חדשה למסד הנתונים
    """
    print(f"📥 קבלת בקשה להוספת frame {entry.frame_number}")

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
        print(f"✅ נוסף frame {entry.frame_number} למסד הנתונים (ID: {db_entry.id})")
        return {"status": "success", "id": db_entry.id}

    except Exception as e:
        db.rollback()
        print(f"❌ שגיאה בהוספת frame {entry.frame_number}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to insert video entry: {str(e)}")


@app.get("/get_all_videos/", response_model=List[VideoEntryResponse])
def get_all_videos(
        limit: Optional[int] = 1000,
        only_with_location: bool = False,
        db: Session = Depends(get_db)
):
    """
    שליפת כל רשומות הוידאו (עם אפשרות לסינון)
    """
    try:
        query = db.query(VideoEntryDB)

        if only_with_location:
            # רק רשומות עם מיקום תקין
            query = query.filter(
                VideoEntryDB.latitude != 0,
                VideoEntryDB.longitude != 0
            )

        # מיון לפי זמן (החדשות ביותר קודם)
        query = query.order_by(desc(VideoEntryDB.id))

        # הגבלת מספר התוצאות
        if limit:
            query = query.limit(limit)

        videos = query.all()
        print(f"📤 החזרת {len(videos)} רשומות וידאו")
        return videos

    except Exception as e:
        print(f"❌ שגיאה בשליפת נתוני וידאו: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch videos: {str(e)}")


@app.get("/get_videos_by_path/{video_path:path}", response_model=List[VideoEntryResponse])
def get_videos_by_path(video_path: str, db: Session = Depends(get_db)):
    """
    שליפת רשומות לפי נתיב וידאו ספציפי
    """
    try:
        videos = db.query(VideoEntryDB).filter(
            VideoEntryDB.video_path == video_path
        ).order_by(VideoEntryDB.frame_number).all()

        print(f"📤 החזרת {len(videos)} רשומות עבור וידאו: {video_path}")
        return videos

    except Exception as e:
        print(f"❌ שגיאה בשליפת נתונים לוידאו {video_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch videos: {str(e)}")


@app.get("/get_recent_videos/", response_model=List[VideoEntryResponse])
def get_recent_videos(minutes: int = 60, db: Session = Depends(get_db)):
    """
    שליפת רשומות מהשעות האחרונות
    """
    try:
        # חישוב זמן תחילת הטווח
        time_threshold = datetime.now() - timedelta(minutes=minutes)

        videos = db.query(VideoEntryDB).filter(
            VideoEntryDB.time >= time_threshold.strftime('%Y-%m-%d %H:%M:%S')
        ).order_by(desc(VideoEntryDB.id)).all()

        print(f"📤 החזרת {len(videos)} רשומות מה-{minutes} דקות האחרונות")
        return videos

    except Exception as e:
        print(f"❌ שגיאה בשליפת רשומות אחרונות: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent videos: {str(e)}")


@app.get("/get_video_stats/", response_model=VideoStats)
def get_video_stats(db: Session = Depends(get_db)):
    """
    קבלת סטטיסטיקות כלליות על הנתונים
    """
    try:
        # סך הכל רשומות
        total_entries = db.query(VideoEntryDB).count()

        # סך כל האנשים שזוהו
        total_people = db.query(func.sum(VideoEntryDB.pedestrian_count)).scalar() or 0

        # מספר וידאוים ייחודיים
        unique_videos = db.query(VideoEntryDB.video_path).distinct().count()

        # רשומות עם מיקום
        entries_with_location = db.query(VideoEntryDB).filter(
            VideoEntryDB.latitude != 0,
            VideoEntryDB.longitude != 0
        ).count()

        # רשומה אחרונה
        latest_entry = db.query(VideoEntryDB).order_by(desc(VideoEntryDB.id)).first()
        latest_time = latest_entry.time if latest_entry else None

        # ממוצע אנשים לפריים
        avg_people = total_people / total_entries if total_entries > 0 else 0

        stats = VideoStats(
            total_entries=total_entries,
            total_people_detected=total_people,
            unique_videos=unique_videos,
            entries_with_location=entries_with_location,
            latest_entry_time=latest_time,
            average_people_per_frame=round(avg_people, 2)
        )

        print(f"📊 סטטיסטיקות: {total_entries} רשומות, {total_people} אנשים")
        return stats

    except Exception as e:
        print(f"❌ שגיאה בחישוב סטטיסטיקות: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate stats: {str(e)}")


@app.put("/update_location/", response_model=dict)
def update_video_location(location_data: LocationUpdate, db: Session = Depends(get_db)):
    """
    עדכון נתוני מיקום בלבד לרשומה קיימת במסד הנתונים
    """
    print(f"🔄 עדכון מיקום עבור frame {location_data.frame_number}")

    # חיפוש הרשומה הקיימת
    existing_entry = db.query(VideoEntryDB).filter(
        VideoEntryDB.frame_number == location_data.frame_number,
        VideoEntryDB.video_path == location_data.video_path
    ).first()

    if not existing_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Video entry not found for frame {location_data.frame_number} in video {location_data.video_path}"
        )

    # עדכון שדות המיקום
    existing_entry.height = location_data.height
    existing_entry.longitude = location_data.longitude
    existing_entry.latitude = location_data.latitude

    try:
        db.commit()
        db.refresh(existing_entry)

        print(f"✅ עודכן מיקום עבור frame {location_data.frame_number}")
        return {
            "status": "success",
            "updated_id": existing_entry.id,
            "frame_number": existing_entry.frame_number,
            "message": "Location data updated successfully"
        }

    except Exception as e:
        db.rollback()
        print(f"❌ שגיאת מסד נתונים: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update database: {str(e)}")


@app.delete("/delete_video/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    """
    מחיקת רשומה ספציפית
    """
    try:
        video = db.query(VideoEntryDB).filter(VideoEntryDB.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video entry not found")

        db.delete(video)
        db.commit()

        print(f"🗑️ נמחקה רשומה {video_id}")
        return {"status": "success", "message": f"Video entry {video_id} deleted"}

    except Exception as e:
        db.rollback()
        print(f"❌ שגיאה במחיקת רשומה {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")


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


@app.get("/")
def root():
    """
    נקודת קצה ראשית עם מידע על ה-API
    """
    return {
        "message": "Drone Video Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /insert_video/": "הוספת רשומת וידאו",
            "GET /get_all_videos/": "שליפת כל הרשומות",
            "GET /get_videos_by_path/{path}": "שליפת רשומות לפי וידאו",
            "GET /get_recent_videos/": "שליפת רשומות אחרונות",
            "GET /get_video_stats/": "סטטיסטיקות כלליות",
            "PUT /update_location/": "עדכון מיקום",
            "DELETE /delete_video/{id}": "מחיקת רשומה"
        }
    }


# הוספת import שחסר
from sqlalchemy import func
from datetime import timedelta

# Main function to run the server
if __name__ == "__main__":
    import uvicorn

    print("🚀 מפעיל שרת API...")
    print("📍 השרת יעבוד על: http://localhost:8080")
    print("📖 תיעוד האפליקציה: http://localhost:8080/docs")


    uvicorn.run("server.api_server:app", host="127.0.0.1", port=8080, reload=True)
