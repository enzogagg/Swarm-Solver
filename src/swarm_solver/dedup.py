import logging

import numpy as np

from .config import Settings
from .llm import LLMClient
from .logs import LOG
from .schemas import GateVerdict, Idea

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


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)


def select_diverse(
    vectors: np.ndarray,
    scores: list[float],
    k: int,
    existing: np.ndarray | None = None,
    diversity: float = 1.0,
) -> list[int]:
    """Choisit `k` idées en équilibrant note et variété (MMR).

    À chaque tour, on prend l'idée qui maximise `note/10 - diversity * ressemblance maximale` avec ce qui est déjà
    choisi ou déjà audité (`existing`). Deux idées de même mécanisme ne partent donc pas ensemble à l'audit, qui est
    l'étape la plus chère. `diversity=0` redonne le simple classement par note.
    """
    n = len(vectors)
    if n == 0 or k <= 0:
        return []
    unit = _unit(vectors)
    closest = np.zeros(n)
    if existing is not None and len(existing):
        closest = np.clip(unit @ _unit(existing).T, 0, 1).max(axis=1)
    chosen: list[int] = []
    remaining = set(range(n))
    while remaining and len(chosen) < k:
        best = max(remaining, key=lambda i: (scores[i] / 10 - diversity * closest[i], scores[i], -i))
        chosen.append(best)
        remaining.remove(best)
        closest = np.maximum(closest, np.clip(unit @ unit[best], 0, 1))
    return chosen


async def pick_diverse(
    llm: LLMClient,
    settings: Settings,
    candidates: list[tuple[Idea, GateVerdict]],
    k: int,
    existing: list[Idea] | None = None,
) -> list[tuple[Idea, GateVerdict]]:
    """Les `k` idées à auditer parmi `candidates`, variées et bien notées, loin de celles déjà auditées (`existing`).

    Si les embeddings sont indisponibles, on retombe sur le classement par note: mieux vaut un choix moins varié
    qu'un run interrompu.
    """
    by_score = sorted(candidates, key=lambda iv: iv[1].score, reverse=True)
    if len(by_score) <= k or settings.diversity <= 0:
        return by_score[:k]
    existing = existing or []
    try:
        vectors = await llm.embed(
            settings.embed_model, [f"{i.title}. {i.description}" for i in [*(i for i, _ in by_score), *existing]]
        )
    except Exception as e:  # noqa: BLE001 - le choix diversifié est un plus, pas une condition
        log.warning("embeddings indisponibles (%s: %s): sélection par note seule", type(e).__name__, e)
        return by_score[:k]
    n = len(by_score)
    picked = select_diverse(
        vectors[:n], [v.score for _, v in by_score], k, vectors[n:] if existing else None, settings.diversity
    )
    picked_set = set(picked)
    log.debug("sélection variée: %d idées sur %d, %d des %d mieux notées écartées pour ressemblance",
              len(picked), n, sum(1 for i in range(k) if i not in picked_set), k)
    return [by_score[i] for i in picked]
