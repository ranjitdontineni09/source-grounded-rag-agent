"""Agent loop: session plus retrieve tool.

Generation is gated on non-empty hits. Empty retrieval refuses without calling
the generator.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agent.chunk import Chunk, load_kb
from agent.generate import generate
from agent.qdrant_store import client as qdrant_client
from agent.qdrant_store import search as qdrant_search
from agent.qdrant_store import upsert as qdrant_upsert
from agent.retrieve import lexical


@dataclass
class Turn:
    """One ask/follow-up in a session.

    Attributes:
        question: User text for this turn.
        answer: Model or refuse message.
        refused: True when retrieval was empty or the model returned REFUSE.
        citations: Source quotes; empty on refuse.
        tool: Retriever that ran (``retrieve_qdrant`` or ``retrieve_lexical``).
    """

    question: str
    answer: str
    refused: bool
    citations: list[dict]
    tool: str


@dataclass
class Session:
    """Conversation that follow-ups attach to.

    Attributes:
        id: Opaque session identifier returned to the client.
        turns: Ordered ask/answer pairs.
    """

    id: str
    turns: list[Turn] = field(default_factory=list)


class Harness:
    """Cite-or-refuse controller around retrieval and generation."""

    def __init__(self, kb: list[Chunk]):
        """Index ``kb`` and optionally upsert into Qdrant.

        Args:
            kb: Chunks loaded from the markdown knowledge base.
        """
        self.kb = kb
        self.sessions: dict[str, Session] = {}
        self.qdrant = qdrant_client()
        if self.qdrant is not None:
            qdrant_upsert(self.qdrant, kb)

    @property
    def backend(self) -> str:
        """Return ``qdrant`` or ``lexical`` depending on connectivity."""
        return "qdrant" if self.qdrant is not None else "lexical"

    def ask(self, question: str, session_id: str | None = None) -> dict:
        """Retrieve, then generate or refuse.

        Args:
            question: User question.
            session_id: Existing session to continue, or ``None`` to start one.

        Returns:
            A dict with ``session_id``, ``refused``, ``answer``, ``citations``,
            and ``tool``.
        """
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
        """Run Qdrant search or lexical overlap.

        Args:
            query: Question, possibly concatenated with the prior turn.

        Returns:
            A tuple ``(hits, tool_name)``.
        """
        if self.qdrant is not None:
            hits = qdrant_search(self.qdrant, query)
            return hits, "retrieve_qdrant"
        return lexical(query, self.kb), "retrieve_lexical"

    def _session(self, session_id: str | None) -> Session:
        """Return an existing session or create one.

        Args:
            session_id: Client-supplied id, or ``None``.

        Returns:
            The session stored on this harness.
        """
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        session = Session(id=session_id or str(uuid.uuid4()))
        self.sessions[session.id] = session
        return session

    def _followup_query(self, session: Session, question: str) -> str:
        """Blend the last question into the retrieve query for follow-ups.

        Args:
            session: Current conversation.
            question: New user text.

        Returns:
            ``question`` alone on the first turn, otherwise prior + current.
        """
        if not session.turns:
            return question
        prior = session.turns[-1].question
        return f"{prior} {question}"

    def _payload(self, session: Session, turn: Turn) -> dict:
        """Build the HTTP-facing turn dict.

        Args:
            session: Conversation that owns ``turn``.
            turn: Latest result.

        Returns:
            JSON-serializable response body.
        """
        return {
            "session_id": session.id,
            "refused": turn.refused,
            "answer": turn.answer,
            "citations": turn.citations,
            "tool": turn.tool,
        }


def boot(kb_dir: Path) -> Harness:
    """Load markdown under ``kb_dir`` and construct a harness.

    Args:
        kb_dir: Directory of ``*.md`` source files.

    Returns:
        A ready :class:`Harness`.
    """
    return Harness(load_kb(kb_dir))
