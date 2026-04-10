"""Utility functions for ResearchOS.

Provides embedding computation, similarity metrics, LLM call interface,
and data loading helpers. All LLM/embedding calls gracefully fall back
to mock mode when API keys are unavailable.
"""

import json
import os
import math
from typing import Optional


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

_embedding_model = None


def _load_embedding_model():
    """Lazy-load sentence-transformers model, return None if unavailable."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _embedding_model
    except Exception:
        return None


def get_embedding(text: str) -> list[float]:
    """Compute an embedding vector for the given text.

    Uses sentence-transformers if available, otherwise returns a
    deterministic mock embedding based on a hash of the text.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = _load_embedding_model()
    if model is not None:
        vec = model.encode(text)
        return vec.tolist()

    # Mock fallback: deterministic 384-dim vector from text hash
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    dim = 384
    values = []
    for i in range(dim):
        byte_pair = h[(2 * i) % len(h): (2 * i + 2) % len(h)] or h[:2]
        values.append((int(byte_pair, 16) - 128) / 128.0)
    # Normalize
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# LLM call interface
# ---------------------------------------------------------------------------

# Set to True to force mock mode even when API keys are present
MOCK_MODE = os.environ.get("RESEARCHOS_MOCK", "0") == "1"


def call_llm(prompt: str, model: str = "claude-sonnet-4-20250514",
             system: Optional[str] = None) -> str:
    """Unified LLM call interface.

    Attempts to call the Anthropic API if an API key is set and mock mode
    is not forced. Falls back to a mock response that echoes the prompt
    for development and testing.

    Args:
        prompt: The user prompt to send.
        model: Model identifier (e.g. 'claude-sonnet-4-20250514').
        system: Optional system prompt.

    Returns:
        The LLM response text.
    """
    if not MOCK_MODE:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                messages = [{"role": "user", "content": prompt}]
                kwargs = {"model": model, "max_tokens": 4096, "messages": messages}
                if system:
                    kwargs["system"] = system
                response = client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                print(f"[ResearchOS] LLM call failed ({e}), falling back to mock mode.")

    # Mock mode
    print(f"[ResearchOS MOCK] call_llm(model={model})")
    print(f"  system: {system[:120] + '...' if system and len(system) > 120 else system}")
    print(f"  prompt: {prompt[:200] + '...' if len(prompt) > 200 else prompt}")
    return f"[MOCK RESPONSE for model={model}] This is a placeholder response. Prompt length: {len(prompt)} chars."


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_papers(path: str) -> list[dict]:
    """Load papers from a JSON file or directory of JSON files.

    Each paper dict is expected to have at least 'id' and 'text' fields.

    Args:
        path: Path to a JSON file or a directory containing JSON files.

    Returns:
        A list of paper dicts.
    """
    papers: list[dict] = []
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                papers.extend(data)
            else:
                papers.append(data)
    elif os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json"):
                with open(os.path.join(path, fname), "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        papers.extend(data)
                    else:
                        papers.append(data)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    return papers
