from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from .database import get_db, NoteDB
from .models import Note, NoteCreate, NoteUpdate

app = FastAPI(
    title="AI Notes Service",
    description="Stage 0: Production-ready REST API. Later stages will add embeddings and semantic search.",
    version="1.0.0",
)


@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — used by Docker and CI to verify the service is up."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/notes", response_model=Note, status_code=201, tags=["Notes"])
def create_note(note: NoteCreate, db: Session = Depends(get_db)):
    """Create a new note."""
    db_note = NoteDB(**note.model_dump())
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note


@app.get("/notes", response_model=List[Note], tags=["Notes"])
def list_notes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all notes with pagination."""
    return db.query(NoteDB).offset(skip).limit(limit).all()


@app.get("/notes/{note_id}", response_model=Note, tags=["Notes"])
def get_note(note_id: int, db: Session = Depends(get_db)):
    """Get a single note by ID."""
    note = db.query(NoteDB).filter(NoteDB.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
    return note


@app.put("/notes/{note_id}", response_model=Note, tags=["Notes"])
def update_note(note_id: int, update: NoteUpdate, db: Session = Depends(get_db)):
    """Update title and/or content of a note."""
    note = db.query(NoteDB).filter(NoteDB.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
    for field, value in update.model_dump(exclude_none=True).items():
        setattr(note, field, value)
    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return note


@app.delete("/notes/{note_id}", status_code=204, tags=["Notes"])
def delete_note(note_id: int, db: Session = Depends(get_db)):
    """Delete a note by ID."""
    note = db.query(NoteDB).filter(NoteDB.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
    db.delete(note)
    db.commit()
