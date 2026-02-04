import asyncio
from random import sample

from ragas.metrics._context_precision import ContextPrecision, NonLLMContextPrecisionWithReference


async def compute_precision(retrieved, refs):
    metric = NonLLMContextPrecisionWithReference()
    result = await metric.ascore(
        retrieved_contexts=retrieved,
        reference_contexts=refs
    )
    print("Non‑LLM Precision:", result.value)


# -----------------------------
# Metric
context_precision = ContextPrecision()


# -----------------------------
# Async scoring function
async def main():
    score = await context_precision.abatch_score(sample)
    print("Context Precision Score:", score)


# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
