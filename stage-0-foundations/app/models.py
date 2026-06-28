from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, example="My first note")
    content: str = Field(..., min_length=1, example="This is the note content")


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)


class Note(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
