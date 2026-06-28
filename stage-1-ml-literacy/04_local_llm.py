"""
STAGE 1 — Script 4: Run a Local LLM
=====================================
Run an AI model completely on YOUR laptop.
No API key. No internet needed. No cost per token.

We use GPT-2 — a small but real language model by OpenAI (open source).
- 124 million parameters
- Runs on CPU (no GPU needed)
- Downloads once (~500MB), then works offline

This teaches you:
- How text generation actually works (next-token prediction)
- What temperature does
- What max_length does
- The difference between greedy (temp=0) and sampling (temp>0)
"""

from transformers import pipeline, set_seed
import warnings
warnings.filterwarnings("ignore")

print("Loading GPT-2 model... (downloads once ~500MB)")
print("This may take 2-3 minutes on first run...\n")

generator = pipeline("text-generation", model="gpt2")
print("Model loaded! Let's generate text.\n")


def generate(prompt, temperature=1.0, max_new_tokens=60, seed=42):
    set_seed(seed)
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=50256,
    )
    return result[0]["generated_text"]


# ── Example 1: Basic generation ────────────────────────────────
print("=" * 60)
print("BASIC TEXT GENERATION")
print("=" * 60)

prompt = "Artificial intelligence is"
output = generate(prompt)
print(f"\nPrompt:  '{prompt}'")
print(f"Output:  '{output}'")

# ── Example 2: Temperature effect ─────────────────────────────
print("\n" + "=" * 60)
print("TEMPERATURE EFFECT — Same prompt, different results")
print("=" * 60)
print("\nLow temperature (0.1) = predictable, focused")
print("High temperature (1.5) = creative, random\n")

prompt2 = "The future of AI will"

temps = [0.1, 0.7, 1.5]
for temp in temps:
    result = generate(prompt2, temperature=temp, seed=42)
    print(f"Temperature {temp}:")
    print(f"  '{result[:120]}...'")
    print()

# ── Example 3: Next-token prediction explained ─────────────────
print("=" * 60)
print("HOW AI GENERATES TEXT — One token at a time")
print("=" * 60)

print("""
The AI NEVER sees the full sentence at once.
It predicts ONE token, then feeds that back as new input.

Step 1: Input: "The cat sat on the"
        Predict next token -> "mat" (probability: 45%)

Step 2: Input: "The cat sat on the mat"
        Predict next token -> "." (probability: 62%)

Step 3: Input: "The cat sat on the mat ."
        Predict next token -> <END> (probability: 78%)

This is AUTOREGRESSIVE generation.
This is why longer outputs take longer — more steps.
This is why streaming shows text word by word.
""")

# ── Example 4: Show token-by-token ────────────────────────────
print("=" * 60)
print("STAGE 1 COMPLETE — What You Learned")
print("=" * 60)
print("""
Script 1 — TOKENIZER:
  Text -> Token IDs (numbers)
  'Hello' -> [15496]
  AI only sees numbers, never raw text

Script 2 — EMBEDDINGS:
  Text -> Vector (384 numbers)
  Similar meaning = similar vector
  Foundation of RAG and semantic search

Script 3 — COSINE SIMILARITY:
  How similar are two texts? -> score from 0.0 to 1.0
  This is exactly how RAG retrieves relevant chunks
  Higher score = more relevant to the question

Script 4 — LOCAL LLM:
  A real AI model running on YOUR laptop
  Next-token prediction = one token at a time
  Temperature controls creativity vs accuracy

NEXT -> Stage 2: Build a structured output extractor
       using real OpenAI/Anthropic API calls
""")

