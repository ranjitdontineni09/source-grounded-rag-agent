# Source-Grounded RAG Agent

FastAPI agent that **retrieves first, then answers**. If Qdrant (or the lexical fallback) returns nothing, the harness **refuses** — the generator never runs. A React UI supports ask, citations, and follow-up in a session.

**Stack:** FastAPI · Llama / LoRA (Hugging Face PEFT) · Qdrant · React

Repo: [github.com/ranjitdontineni09/source-grounded-rag-agent](https://github.com/ranjitdontineni09/source-grounded-rag-agent)

## Architecture

```
React ──POST /api/ask──► harness (session + tools)
                              │
                              ├─ tool: retrieve  → Qdrant cosine (hashed n-gram vectors)
                              │                    lexical fallback if Qdrant is down
                              │
                              ├─ empty hits?  REFUSE  (no LLM call)
                              └─ else generate only from those chunks + cite
```

## Design decisions

| Choice | Why |
|---|---|
| **Retrieve is a tool, not a prompt prefix** | The harness decides whether generation is allowed. An LLM cannot “talk its way” past an empty index. |
| **Refuse on empty retrieval** | Grounding is a control-flow rule, not an instruction the model might ignore. |
| **Qdrant for vectors, lexical fallback for tests** | Clone-and-run and CI stay honest without a GPU. `docker compose` is the vector path. |
| **Hashed n-gram embeddings by default** | Real cosine search in Qdrant with no Torch download. Swap `agent/embed.py` for a Hugging Face encoder when you want semantic RAG. |
| **LoRA as a separate train job** | `lora/train.py` fine-tunes an open-weight Llama-family model on cite-or-refuse pairs (PEFT). The API loads an adapter only if `LORA_PATH` is set. |
| **Sessions for follow-up** | Follow-ups reuse `session_id` so the next retrieve includes prior question context. |

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn agent.app:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) (API + built-in UI). React dev server:

```bash
cd frontend && npm install && npm run dev
```

Vector path:

```bash
docker compose up --build
```

```bash
curl -s http://127.0.0.1:8000/api/ask -H 'content-type: application/json' \
  -d '{"question":"What does the Kafka task mesh do?"}'
```

Ask something outside the KB (`"Who won the 2014 World Cup?"`) — `"refused": true`.

## LoRA

```bash
python lora/train.py --dry-run          # validates dataset (CI)
python lora/train.py --output-dir lora/adapter   # needs GPU + requirements-lora.txt
```

Set `LORA_PATH=lora/adapter` and `LLM_MODEL=...` to serve the adapter. Without them the runtime still cites or refuses using the grounded synthesizer.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/ask` | `{ "question", "session_id?" }` → answer or refuse + citations |
| `GET` | `/api/session/{id}` | Follow-up history |
| `GET` | `/health` | Docs indexed + retriever backend |
