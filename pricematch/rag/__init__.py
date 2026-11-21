"""Retrieval-augmented generation (RAG) agent for PriceMatch.

Pipeline: the tracked catalog (offers + price history) is turned into a
small document corpus, a retriever ranks those documents for the user's
question, and a generator writes an answer grounded ONLY in the ranked
documents, citing them as [1], [2], ...

Both stages have a neural backend (needs requirements-ml.txt: MiniLM
embeddings + FLAN-T5 generation) and a zero-dependency offline backend
(TF-IDF retrieval + grounded template generation), so the demo never
breaks: same architecture, swappable parts.
"""
