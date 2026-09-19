import logging

import numpy as np

from .config import Settings
from .llm import LLMClient
from .logs import LOG
from .schemas import Idea

log = logging.getLogger(f"{LOG}.dedup")


def dedupe_indices(vectors: np.ndarray, threshold: float, fixed: int = 0) -> list[int]:
    """Sélection gloutonne : garde un vecteur seulement s'il est assez éloigné de tous ceux déjà gardés.

    Les `fixed` premiers vecteurs sont des références déjà acceptées : elles servent de comparaison
    mais ne sont pas retournées.
    """
    if len(vectors) == 0:
        return []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.clip(norms, 1e-9, None)
    kept: list[int] = []
    kept_matrix = np.empty_like(unit)
    count = 0
    for i, v in enumerate(unit):
        if i >= fixed and count and float(np.max(kept_matrix[:count] @ v)) >= threshold:
            continue
        kept_matrix[count] = v
        count += 1
        if i >= fixed:
            kept.append(i)
    return kept


async def dedupe_ideas(
    llm: LLMClient, settings: Settings, ideas: list[Idea], existing: list[Idea] | None = None
) -> list[Idea]:
    """Retire les doublons entre `ideas`, et par rapport aux idées `existing` déjà connues."""
    if not ideas:
        return []
    existing = existing or []
    texts = [f"{i.title}. {i.description}" for i in [*existing, *ideas]]
    vectors = await llm.embed(settings.embed_model, texts)
    kept = dedupe_indices(vectors, settings.dedup_threshold, fixed=len(existing))
    log.debug("dédup: %d -> %d idées (seuil %.2f)", len(ideas), len(kept), settings.dedup_threshold)
    return [ideas[i - len(existing)] for i in kept]
