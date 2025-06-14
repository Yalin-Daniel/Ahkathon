from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import List
import os

app = FastAPI()

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
    time = Column(String, default=datetime.utcnow)
    video_path = Column(String, default=datetime.utcnow)
    height = Column(Integer),
    longitude = Column(Float),
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
# class BoundingBox(BaseModel):
#     x: int
#     y: int
#     width: int
#     height: int

# class PersonInfo(BaseModel):
#     detection_id: int
#     confidence: float
#     frame_number: int
#     # bounding_box: BoundingBox

class VideoEntry(BaseModel):
    frame_number: int
    timestamp: float
    pedestrian_count: int
    time: str
    video_path: str
    height: int
    longitude = float
    latitude = float
    # detections: List[PersonInfo]

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


# Video endpoints
@app.post("/insert_video/")
def insert_video(entry: VideoEntry, db: Session = Depends(get_db)):
    print("got request")
    db_entry = VideoEntryDB(
        frame_number=entry.frame_number,
        timestamp=entry.timestamp,
        pedestrian_count=entry.pedestrian_count,
        time=entry.time,
        video_path=entry.video_path,
        height = 0,
        longitude = 0,
        latitude = 0
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)

    print(f"Added frame {entry.frame_number} to db")
    return {"status": "success", "id": db_entry.id}


# User endpoints
@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user with this email already exists
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


# Main function to run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)