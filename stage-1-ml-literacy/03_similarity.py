"""
STAGE 1 — Script 3: Cosine Similarity
=======================================
What is cosine similarity?
- A number between 0 and 1 that measures HOW SIMILAR two texts are
- 1.0 = identical meaning
- 0.0 = completely unrelated
- 0.5+ = somewhat related

This is EXACTLY what happens inside a RAG system:
  User question -> embed it -> compare with all document chunks
  -> return chunks with highest similarity score
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(vec1, vec2):
    """Measure similarity between two vectors. Returns 0.0 to 1.0"""
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude


def compare(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    score = cosine_similarity(emb1, emb2)
    bar = "#" * int(score * 30)
    print(f"  '{text1}'")
    print(f"  '{text2}'")
    print(f"  Similarity: {score:.3f}  {bar}")
    print()


# ── Example 1: Obvious comparisons ────────────────────────────
print("=" * 60)
print("SIMILARITY SCORES — Obvious Cases")
print("=" * 60)
print()

compare("I love pizza", "I love pizza")
compare("I love pizza", "Pizza is my favourite food")
compare("I love pizza", "I enjoy Italian cuisine")
compare("I love pizza", "The weather is nice today")
compare("I love pizza", "Stock markets crashed")

# ── Example 2: RAG simulation ──────────────────────────────────
print("=" * 60)
print("RAG SIMULATION — Find the most relevant document chunk")
print("=" * 60)

user_question = "How do I stay healthy?"

document_chunks = [
    "Exercise regularly and eat nutritious food to stay fit",
    "The capital of France is Paris",
    "Drinking water and sleeping 8 hours improves your health",
    "Python is a popular programming language for AI",
    "Meditation and yoga reduce stress and improve wellbeing",
    "The stock market had record gains this quarter",
    "Eating vegetables and fruits prevents many diseases",
]

print(f"\nUser question: '{user_question}'")
print(f"\nSearching {len(document_chunks)} document chunks...\n")

question_embedding = model.encode(user_question)
results = []

for chunk in document_chunks:
    chunk_embedding = model.encode(chunk)
    score = cosine_similarity(question_embedding, chunk_embedding)
    results.append((score, chunk))

# Sort by similarity (highest first)
results.sort(reverse=True)

print("RANKED RESULTS (highest similarity first):")
print("-" * 50)
for rank, (score, chunk) in enumerate(results, 1):
    bar = "#" * int(score * 25)
    marker = " <- TOP RESULT" if rank == 1 else ""
    print(f"  #{rank}  {score:.3f}  {bar}")
    print(f"       '{chunk}'{marker}")
    print()

print("=" * 60)
print("KEY LESSON:")
print("  This ranked list IS how RAG retrieves documents!")
print("  Top 3-5 results get injected into the LLM prompt.")
print("  LLM answers using ONLY those retrieved chunks.")
print("  That's why RAG doesn't hallucinate — it reads YOUR docs.")
print("=" * 60)

