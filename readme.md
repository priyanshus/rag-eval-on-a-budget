### rag-eval-on-a-budget
Playground repository to develop and evaluate a RAG application against structured data set.

## How to run
- Create a .env file, provide below info:
```commandline
OPENROUTER_API_KEY=<open-router-api-key>
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
LUNARY_PUBLIC_KEY=<lunary-public-key>
TOKENIZERS_PARALLELISM=False
```
- Run rag.py

## Evaluation Strategy

```commandline
Tier 0: Data sanity  ──┐
Tier 1: Recall@k     ├──> FAIL FAST (cheap)
Tier 2: Robustness   ──┘
        ↓
Tier 3: Heuristics (cheap filters)
        ↓
Tier 4: LLM Judge (selective)
        ↓
Tier 5: Human review (rare)
```