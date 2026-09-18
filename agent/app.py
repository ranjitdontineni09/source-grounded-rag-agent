from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.harness import boot

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "frontend" / "dist"
app = FastAPI(title="Source-Grounded RAG Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
harness = boot(ROOT / "kb")


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=800)
    session_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "docs": len({c.doc for c in harness.kb}),
        "chunks": len(harness.kb),
        "retriever": harness.backend,
    }


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    return harness.ask(body.question, body.session_id)


@app.get("/api/session/{session_id}")
def session(session_id: str) -> dict:
    found = harness.sessions.get(session_id)
    if not found:
        raise HTTPException(status_code=404, detail="unknown session")
    return {
        "session_id": found.id,
        "turns": [
            {
                "question": t.question,
                "answer": t.answer,
                "refused": t.refused,
                "citations": t.citations,
            }
            for t in found.turns
        ],
    }


if UI.is_dir():
    app.mount("/assets", StaticFiles(directory=UI / "assets"), name="assets")

    @app.get("/")
    def spa() -> FileResponse:
        return FileResponse(UI / "index.html")
else:

    @app.get("/")
    def fallback_ui():
        from fastapi.responses import HTMLResponse

        return HTMLResponse(
            """<!doctype html><meta charset="utf-8"><title>Source-Grounded RAG Agent</title>
            <p>API is up. Build the React UI with <code>cd frontend && npm install && npm run build</code>
            or run <code>npm run dev</code> on port 5173.</p>"""
        )
