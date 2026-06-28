# Stage 1 — ML Literacy: Tokens, Embeddings & Local LLM

> PDF Roadmap: "Use Hugging Face to run a small open model, tokenize text,
> generate embeddings and compute cosine similarity between sentences."

## What This Stage Teaches

| Script | Concept | What You See |
|--------|---------|-------------|
| `01_tokenizer.py` | Tokens | Text → numbers, cost calculation |
| `02_embeddings.py` | Embeddings | Text → 384-number vector |
| `03_similarity.py` | Cosine Similarity | RAG retrieval simulation |
| `04_local_llm.py` | Local LLM | GPT-2 running on your laptop |

## Run Order

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Run each script in order
python 01_tokenizer.py
python 02_embeddings.py
python 03_similarity.py
python 04_local_llm.py   # downloads GPT-2 ~500MB first time
```

## Key Concepts Learned

**Tokens** — AI reads numbers not words. Every token costs money.

**Embeddings** — Text converted to 384 numbers. Similar meaning = similar numbers. Foundation of ALL RAG systems.

**Cosine Similarity** — Score from 0.0 to 1.0. How RAG ranks and retrieves documents.

**Local LLM** — AI running on your own machine. Next-token prediction, temperature, autoregressive generation.

## Interview Talking Points
- "Embeddings are like GPS coordinates for meaning — similar sentences cluster together in vector space."
- "RAG works by embedding both the question and all document chunks, then finding the closest chunks using cosine similarity."
- "Temperature controls the probability distribution — low temp peaks sharply at the most likely token, high temp flattens the distribution making rare tokens more likely."
- "Every LLM generates one token at a time — that's why streaming exists and why longer outputs take longer."
