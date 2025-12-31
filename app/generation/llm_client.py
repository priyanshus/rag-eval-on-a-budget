import os
from typing import List, Optional, Dict

import litellm
from litellm import completion


class LLMClient:
    def __init__(
            self,
            model: str,
            temperature: float = 0.2,
            max_tokens: int = 512,
    ):

        self._api_key = os.environ["OPENROUTER_API_KEY"]
        self._lunary_public_key = os.environ["LUNARY_PUBLIC_KEY"]
        self._api_base = os.environ["OPENROUTER_API_BASE"]

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_messages(
            self,
            question: str,
            context_chunks: List[str],
            system_prompt: Optional[str] = None,
    ) -> List[Dict]:

        messages = []

        messages.append(
            {
                "role": "system",
                "content": system_prompt
                           or "Answer only using the provided context. If the answer is not present, say you don't know. If context metadata is available, please mention the author, title and link of context.",
            }
        )

        for i, chunk in enumerate(context_chunks):
            messages.append(
                {
                    "role": "system",
                    "content": f"Context {i + 1}: {chunk['text']} metadata: {chunk['metadata']}",
                }
            )

        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        return messages

    def generate(
            self,
            question: str,
            context_chunks: List[str],
            system_prompt: Optional[str] = None,
            stream: bool = False,
    ):
        """
        Run the RAG query.
        """

        messages = self._build_messages(
            question=question,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
        )

        litellm.success_callback = ["lunary"]

        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
            api_key=self._api_key,
            base_url=self._api_base
        )

        if stream:
            return response
        else:
            return response["choices"][0]["message"]["content"]
