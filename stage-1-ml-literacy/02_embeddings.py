"""
STAGE 1 — Script 2: Embeddings
================================
What are embeddings?
- Embeddings convert text into a LIST OF NUMBERS (a vector)
- Similar meanings -> similar numbers
- This is the foundation of RAG, semantic search, and GraphRAG

Think of it like GPS coordinates:
  Mumbai  -> (19.07, 72.87)
  Delhi   -> (28.70, 77.10)
  London  -> (51.50, -0.12)

  Mumbai and Delhi are closer to each other than to London.
  Embeddings do the same — but for MEANING, not geography.
  And instead of 2 numbers, we use 384 numbers.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model... (downloads once, ~90MB)")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

# ── Example 1: See what an embedding looks like ────────────────
print("=" * 60)
print("WHAT DOES AN EMBEDDING LOOK LIKE?")
print("=" * 60)

text = "I love learning about AI"
embedding = model.encode(text)

print(f"\nText: '{text}'")
print(f"Embedding shape: {embedding.shape}  <- 384 numbers")
print(f"First 10 numbers: {embedding[:10].round(4)}")
print(f"Last  10 numbers: {embedding[-10:].round(4)}")
print(f"\nEvery sentence becomes exactly 384 numbers.")
print(f"Similar sentences -> similar 384 numbers.")

# ── Example 2: Similar vs different sentences ──────────────────
print("\n" + "=" * 60)
print("SIMILAR VS DIFFERENT SENTENCES")
print("=" * 60)

sentences = [
    "I love pizza",
    "Pizza is my favourite food",       # very similar to above
    "I enjoy eating Italian food",      # somewhat similar
    "The stock market crashed today",   # completely different
    "Machine learning is fascinating",  # different topic
]

embeddings = model.encode(sentences)

print(f"\nGenerated {len(sentences)} embeddings")
print(f"Each embedding: {embeddings[0].shape[0]} dimensions")

for i, (s, e) in enumerate(zip(sentences, embeddings)):
    print(f"\n  [{i+1}] '{s}'")
    print(f"       First 5 numbers: {e[:5].round(3)}")

print("\n" + "=" * 60)
print("KEY LESSON:")
print("  Text -> Embedding (384 numbers)")
print("  Similar text -> Similar numbers")
print("  This is how RAG finds 'relevant' documents!")
print("=" * 60)

