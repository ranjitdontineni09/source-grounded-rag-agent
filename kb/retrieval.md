# Retrieval and refuse

The harness calls retrieve before it generates. Hits come from Qdrant cosine search on chunk embeddings, or a lexical fallback when Qdrant is down.

If retrieval returns nothing, the agent refuses and the generator never runs. A citation is a document title plus a short quote from the chunk.

Follow-up questions reuse the session id so the next retrieve includes the prior question.
