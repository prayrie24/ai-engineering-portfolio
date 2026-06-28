"""
STAGE 1 — Script 2 (Lite Version): Embeddings
================================================
This version works WITHOUT torch/sentence-transformers.
Uses numpy only — teaches the CONCEPT of embeddings clearly.

Real embeddings use neural networks with 384+ dimensions.
This demo uses simplified 5-dimension vectors to SHOW THE IDEA.
"""

import numpy as np

print("=" * 60)
print("WHAT IS AN EMBEDDING? (Simplified Demo)")
print("=" * 60)

print("""
An embedding converts text into a list of numbers (vector).
Similar meaning = similar numbers.

Think of it like rating a sentence on 5 scales:
  [food, tech, health, emotion, complexity]
  Each scale goes from -1.0 (low) to +1.0 (high)
""")

# Simplified hand-crafted embeddings for demo
# Real embeddings have 384-1536 dimensions, learned by a neural network
# These 5D ones show the SAME concept clearly

embeddings = {
    "I love pizza":                    np.array([0.9,  0.0,  0.2,  0.8,  0.1]),
    "Pizza is my favourite food":      np.array([0.8,  0.0,  0.1,  0.7,  0.2]),
    "I enjoy eating Italian cuisine":  np.array([0.7,  0.0,  0.2,  0.6,  0.3]),
    "Exercise keeps you healthy":      np.array([0.1,  0.0,  0.9,  0.5,  0.3]),
    "Python is great for AI":          np.array([-0.1, 0.9,  0.0,  0.3,  0.7]),
    "Machine learning is fascinating": np.array([0.0,  0.9,  0.1,  0.6,  0.8]),
    "Stock market crashed today":      np.array([0.0,  0.4, -0.1, -0.3,  0.5]),
}

dims = ["food", "tech", "health", "emotion", "complexity"]

print("=" * 60)
print("EMBEDDINGS — Each sentence as 5 numbers")
print(f"  Dimensions: {dims}")
print("=" * 60)

for sentence, vector in embeddings.items():
    print(f"\n  '{sentence}'")
    print(f"   -> {vector}")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("\n" + "=" * 60)
print("SIMILARITY — How close are these sentences?")
print("=" * 60)

query = "I love pizza"
query_vec = embeddings[query]
print(f"\nQuery: '{query}'\n")

results = []
for sentence, vector in embeddings.items():
    if sentence == query:
        continue
    score = cosine_similarity(query_vec, vector)
    results.append((score, sentence))

results.sort(reverse=True)

for score, sentence in results:
    bar = "#" * int(score * 25) + "-" * (25 - int(score * 25))
    print(f"  {score:.3f}  {bar}  '{sentence}'")

print("\n" + "=" * 60)
print("KEY INSIGHT:")
print("  Food sentences cluster together (high food score)")
print("  Tech sentences cluster together (high tech score)")
print("  Real embeddings do this with 384 dimensions")
print("  learned from billions of sentences by a neural network")
print("=" * 60)

