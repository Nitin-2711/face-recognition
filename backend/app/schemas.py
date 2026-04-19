from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    name: str
    employee_id: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    image_path: str
    created_at: datetime

    class Config:
        from_attributes = True

class AttendanceBase(BaseModel):
    status: str
    timestamp: datetime

class Attendance(AttendanceBase):
    id: int
    user_id: Optional[int]
    unknown_face_image: Optional[str]
    user: Optional[User]

    class Config:
        from_attributes = True
