"""
STAGE 1 — Script 1: Tokenizer
==============================
What are tokens?
- AI doesn't read words. It reads numbers called tokens.
- Every word/character gets converted to a number before entering the AI.
- This is WHY AI costs money — you pay per token, not per word.

Run this file to SEE tokens in action.
"""

import tiktoken

# This is the same tokenizer ChatGPT uses
tokenizer = tiktoken.get_encoding("cl100k_base")

print("=" * 60)
print("TOKENIZER — Text becomes numbers")
print("=" * 60)

# ── Example 1: Basic sentence ──────────────────────────────────
text1 = "Hello, I am learning AI from scratch!"
tokens1 = tokenizer.encode(text1)
print(f"\nText:   '{text1}'")
print(f"Tokens: {tokens1}")
print(f"Count:  {len(tokens1)} tokens")

# Decode back to see each token as text
print("\nEach token decoded:")
for i, token_id in enumerate(tokens1):
    word = tokenizer.decode([token_id])
    print(f"  Token {i+1}: ID={token_id:6d}  -> '{word}'")

# ── Example 2: Same meaning, different tokens ──────────────────
print("\n" + "=" * 60)
print("SAME MEANING — DIFFERENT TOKEN COUNT")
print("=" * 60)

sentences = [
    "Hi",
    "Hello",
    "Hello there",
    "Good morning",
    "Quantum entanglement",
    "AI",
    "Artificial Intelligence",
]

for s in sentences:
    t = tokenizer.encode(s)
    print(f"  '{s}' -> {len(t)} token(s) -> {t}")

# ── Example 3: Numbers are bad for AI ─────────────────────────
print("\n" + "=" * 60)
print("WHY AI IS BAD AT MATH — Numbers split into weird tokens")
print("=" * 60)

numbers = ["100", "1000", "10000", "123456789"]
for n in numbers:
    t = tokenizer.encode(n)
    print(f"  '{n}' -> {len(t)} token(s) -> {t}")

# ── Example 4: Cost calculator ────────────────────────────────
print("\n" + "=" * 60)
print("TOKEN COST CALCULATOR")
print("=" * 60)

long_text = """
Artificial intelligence is the simulation of human intelligence
processes by computer systems. These processes include learning,
reasoning, and self-correction. AI is used in many applications
like chatbots, image recognition, and recommendation systems.
"""

tokens = tokenizer.encode(long_text)
token_count = len(tokens)

# GPT-4o pricing (as of 2026)
input_cost_per_1m  = 2.50   # $2.50 per 1M input tokens
output_cost_per_1m = 10.00  # $10.00 per 1M output tokens

cost = (token_count / 1_000_000) * input_cost_per_1m

print(f"\nText length:  {len(long_text)} characters")
print(f"Token count:  {token_count} tokens")
print(f"GPT-4o cost:  ${cost:.6f} (for this one message)")
print(f"1000 messages like this = ${cost * 1000:.4f}")

print("\n" + "=" * 60)
print("KEY LESSON:")
print("  Text -> Tokens (numbers) -> this is ALL the AI sees")
print("  More tokens = more cost + slower response")
print("  Good prompts are SHORT but HIGH SIGNAL")
print("=" * 60)

