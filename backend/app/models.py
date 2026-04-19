from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    encoding = Column(LargeBinary) # Storing face encoding as blob
    created_at = Column(DateTime, default=datetime.utcnow)
    
    attendance = relationship("Attendance", back_populates="user")

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    mask_status = Column(String) # "Mask", "No Mask"
    confidence = Column(Float)
    screenshot_path = Column(String)
    
    user = relationship("User", back_populates="attendance")
