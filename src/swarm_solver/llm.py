import asyncio
import logging
import time
from typing import Protocol, TypeVar

import numpy as np
from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from .config import Settings
from .logs import LOG, TRACE

T = TypeVar("T", bound=BaseModel)

EMBED_BATCH = 128

log = logging.getLogger(f"{LOG}.llm")
trace = logging.getLogger(TRACE)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    async def structured(
        self,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        temperature: float = 0.3,
        seed: int | None = None,
        tag: str = "",
    ) -> T: ...

    async def text(self, model: str, system: str, user: str, *, temperature: float = 0.4, tag: str = "") -> str: ...

    async def embed(self, model: str, texts: list[str]) -> np.ndarray: ...


class OllamaLLM:
    """Client Ollama asynchrone. Le sémaphore borne les requêtes simultanées envoyées au serveur."""

    def __init__(self, settings: Settings, retries: int = 2):
        self._client = AsyncClient(host=settings.host)
        self._sem = asyncio.Semaphore(settings.concurrency)
        self._num_ctx = settings.num_ctx
        self._retries = retries

    def _options(self, temperature: float, seed: int | None) -> dict:
        opts: dict = {"temperature": temperature, "num_ctx": self._num_ctx}
        if seed is not None:
            opts["seed"] = seed
        return opts

    async def _chat(
        self, model: str, system: str, user: str, options: dict, tag: str, fmt: dict | None = None
    ) -> str:
        queued = time.perf_counter()
        async with self._sem:
            started = time.perf_counter()
            resp = await self._client.chat(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=fmt,
                think=False,
                options=options,
                keep_alive="30m",
            )
        done = time.perf_counter()
        content = resp.message.content or ""
        log.debug(
            "%s | %s | attente %.1fs, génération %.1fs | tokens in/out %s/%s | temp %.2f",
            tag, model, started - queued, done - started, resp.prompt_eval_count, resp.eval_count,
            options["temperature"],
        )
        trace.debug("[%s] SYSTEM:\n%s\n\nUSER:\n%s\n\nRESPONSE:\n%s\n%s", tag, system, user, content, "-" * 60)
        return content

    async def structured(
        self,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        temperature: float = 0.3,
        seed: int | None = None,
        tag: str = "",
    ) -> T:
        tag = tag or schema.__name__
        options = self._options(temperature, seed)
        last: Exception | None = None
        for attempt in range(1, self._retries + 2):
            raw = await self._chat(model, system, user, options, tag, schema.model_json_schema())
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                last = e
                log.debug("%s | sortie invalide (essai %d/%d): %s", tag, attempt, self._retries + 1, e)
        raise LLMError(f"Sortie invalide pour {schema.__name__} après {self._retries + 1} essais: {last}")

    async def text(self, model: str, system: str, user: str, *, temperature: float = 0.4, tag: str = "") -> str:
        return await self._chat(model, system, user, self._options(temperature, None), tag or "text")

    async def embed(self, model: str, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        started = time.perf_counter()
        for i in range(0, len(texts), EMBED_BATCH):
            async with self._sem:
                resp = await self._client.embed(model=model, input=texts[i : i + EMBED_BATCH], keep_alive="30m")
            vectors.extend(resp.embeddings)
        log.debug("embed | %s | %d textes en %.1fs", model, len(texts), time.perf_counter() - started)
        return np.asarray(vectors, dtype=np.float32)
