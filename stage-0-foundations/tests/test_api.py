import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

# Use a separate in-memory database for tests — never pollute production db
TEST_DATABASE_URL = "sqlite:///./test_notes.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
Base.metadata.create_all(bind=engine)

client = TestClient(app)


# ── Health ──────────────────────────────────────────────────────────────────

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "timestamp" in response.json()


# ── Create ───────────────────────────────────────────────────────────────────

def test_create_note_success():
    response = client.post("/notes", json={"title": "AI Study", "content": "RAG is powerful"})
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "AI Study"
    assert data["content"] == "RAG is powerful"
    assert "id" in data
    assert "created_at" in data


def test_create_note_empty_title_fails():
    response = client.post("/notes", json={"title": "", "content": "Some content"})
    assert response.status_code == 422


def test_create_note_missing_fields_fails():
    response = client.post("/notes", json={"title": "Only title"})
    assert response.status_code == 422


# ── List ─────────────────────────────────────────────────────────────────────

def test_list_notes_returns_list():
    response = client.get("/notes")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_notes_pagination():
    response = client.get("/notes?skip=0&limit=2")
    assert response.status_code == 200
    assert len(response.json()) <= 2


# ── Get ──────────────────────────────────────────────────────────────────────

def test_get_note_success():
    create = client.post("/notes", json={"title": "Get Test", "content": "Find me"})
    note_id = create.json()["id"]
    response = client.get(f"/notes/{note_id}")
    assert response.status_code == 200
    assert response.json()["id"] == note_id


def test_get_note_not_found():
    response = client.get("/notes/99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ── Update ───────────────────────────────────────────────────────────────────

def test_update_note_title():
    create = client.post("/notes", json={"title": "Old Title", "content": "Same content"})
    note_id = create.json()["id"]
    response = client.put(f"/notes/{note_id}", json={"title": "New Title"})
    assert response.status_code == 200
    assert response.json()["title"] == "New Title"
    assert response.json()["content"] == "Same content"


def test_update_note_content():
    create = client.post("/notes", json={"title": "Same Title", "content": "Old content"})
    note_id = create.json()["id"]
    response = client.put(f"/notes/{note_id}", json={"content": "New content"})
    assert response.status_code == 200
    assert response.json()["content"] == "New content"


def test_update_note_not_found():
    response = client.put("/notes/99999", json={"title": "Ghost"})
    assert response.status_code == 404


# ── Delete ───────────────────────────────────────────────────────────────────

def test_delete_note_success():
    create = client.post("/notes", json={"title": "Delete Me", "content": "Goodbye"})
    note_id = create.json()["id"]
    response = client.delete(f"/notes/{note_id}")
    assert response.status_code == 204
    get = client.get(f"/notes/{note_id}")
    assert get.status_code == 404


def test_delete_note_not_found():
    response = client.delete("/notes/99999")
    assert response.status_code == 404
