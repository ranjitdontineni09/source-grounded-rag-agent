import { useState } from "react";

const STARTERS = [
  "What does the Kafka task mesh do?",
  "When must the agent refuse?",
  "Who won the 2014 World Cup?",
];

export default function App() {
  const [question, setQuestion] = useState(STARTERS[0]);
  const [sessionId, setSessionId] = useState(null);
  const [turns, setTurns] = useState([]);
  const [busy, setBusy] = useState(false);

  async function ask(text) {
    const q = (text ?? question).trim();
    if (q.length < 3) return;
    setBusy(true);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: q, session_id: sessionId }),
      });
      const body = await res.json();
      setSessionId(body.session_id);
      setTurns((prev) => [...prev, { question: q, ...body }]);
      setQuestion("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <p className="kicker">cite or refuse</p>
        <h1>Source-Grounded RAG Agent</h1>
        <p className="lede">
          Retrieval runs first. Empty hits skip the generator. Follow-ups stay in one session.
        </p>
      </header>

      <div className="starters">
        {STARTERS.map((s) => (
          <button key={s} type="button" className="chip" onClick={() => ask(s)} disabled={busy}>
            {s}
          </button>
        ))}
      </div>

      <ol className="thread">
        {turns.map((t, i) => (
          <li key={i} className={t.refused ? "refused" : "ok"}>
            <p className="q">{t.question}</p>
            <p className="a">{t.answer}</p>
            {t.citations?.length > 0 && (
              <ul className="cites">
                {t.citations.map((c, j) => (
                  <li key={j}>
                    <strong>{c.doc}</strong>
                    <span>{c.quote}</span>
                  </li>
                ))}
              </ul>
            )}
            {t.refused && <p className="flag">refused — no retrieved sources</p>}
          </li>
        ))}
      </ol>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask();
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={sessionId ? "Follow-up…" : "Ask the knowledge base…"}
          disabled={busy}
        />
        <button type="submit" disabled={busy}>
          {sessionId ? "Follow up" : "Ask"}
        </button>
      </form>
    </main>
  );
}
