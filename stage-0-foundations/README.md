# Stage 0 — Notes REST API

> PDF Roadmap: "Build and Dockerize a small REST API with tests and CI.
> This is your 'can I ship software at all?' gate."

## What This Project Proves
- Can write clean Python with FastAPI and SQLAlchemy
- Understands REST API design (CRUD, HTTP verbs, status codes)
- Writes real tests (not just happy-path)
- Ships with Docker
- Has CI that blocks broken builds

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/notes` | Create a note |
| GET | `/notes` | List all notes |
| GET | `/notes/{id}` | Get one note |
| PUT | `/notes/{id}` | Update a note |
| DELETE | `/notes/{id}` | Delete a note |

## Tech Stack
- **FastAPI** — modern Python web framework
- **SQLAlchemy** — ORM for SQLite
- **Pydantic v2** — input validation
- **Pytest** — testing
- **Docker** — containerization
- **GitHub Actions** — CI/CD pipeline

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload

# Open docs
# http://localhost:8000/docs
```

## Run Tests

```bash
pytest tests/ -v
```

## Run with Docker

```bash
docker build -t ai-notes-service .
docker run -p 8000:8000 ai-notes-service
```

## Why This Matters for AI (Stage 3+ Preview)
In Stage 3, this same API will gain:
- `/notes/search` — semantic search using embeddings
- `/notes/{id}/similar` — find similar notes
- Vector storage via ChromaDB

The REST foundation you build here is reused across all future stages.

## Interview Talking Points
- "I started with a production-quality API before adding AI — because AI on top of bad software is still bad software."
- "I used dependency injection (`Depends`) to swap the test database without touching production code."
- "Every endpoint has both happy-path and error-case tests."
- "CI fails the build if tests fail — before Docker even tries to build."
