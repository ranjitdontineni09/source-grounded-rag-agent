"""Agent loop: session + retrieve tool. Generation is gated on non-empty hits."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from agent.chunk import Chunk, load_kb
from agent.generate import generate
from agent.qdrant_store import client as qdrant_client
from agent.qdrant_store import search as qdrant_search
from agent.qdrant_store import upsert as qdrant_upsert
from agent.retrieve import lexical


@dataclass
class Turn:
    question: str
    answer: str
    refused: bool
    citations: list[dict]
    tool: str


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)


class Harness:
    def __init__(self, kb: list[Chunk]):
        self.kb = kb
        self.sessions: dict[str, Session] = {}
        self.qdrant = qdrant_client()
        if self.qdrant is not None:
            qdrant_upsert(self.qdrant, kb)

    @property
    def backend(self) -> str:
        return "qdrant" if self.qdrant is not None else "lexical"

    def ask(self, question: str, session_id: str | None = None) -> dict:
        session = self._session(session_id)
        query = self._followup_query(session, question)
        hits, tool = self._retrieve(query)
        if not hits:
            turn = Turn(
                question=question,
                answer="I do not have sources for that.",
                refused=True,
                citations=[],
                tool=tool,
            )
            session.turns.append(turn)
            return self._payload(session, turn)

        text = generate(question, hits, [t.__dict__ for t in session.turns])
        refused = text.strip().upper() == "REFUSE"
        citations = [
            {"doc": h.doc, "quote": h.text[:220], "score": round(h.score, 3)} for h in hits
        ]
        turn = Turn(
            question=question,
            answer="I do not have sources for that." if refused else text,
            refused=refused,
            citations=[] if refused else citations,
            tool=tool,
        )
        session.turns.append(turn)
        return self._payload(session, turn)

    def _retrieve(self, query: str) -> tuple[list[Chunk], str]:
        if self.qdrant is not None:
            hits = qdrant_search(self.qdrant, query)
            return hits, "retrieve_qdrant"
        return lexical(query, self.kb), "retrieve_lexical"

    def _session(self, session_id: str | None) -> Session:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        session = Session(id=session_id or str(uuid.uuid4()))
        self.sessions[session.id] = session
        return session

    def _followup_query(self, session: Session, question: str) -> str:
        if not session.turns:
            return question
        prior = session.turns[-1].question
        return f"{prior} {question}"

    def _payload(self, session: Session, turn: Turn) -> dict:
        return {
            "session_id": session.id,
            "refused": turn.refused,
            "answer": turn.answer,
            "citations": turn.citations,
            "tool": turn.tool,
        }


def boot(kb_dir) -> Harness:
    return Harness(load_kb(kb_dir))
