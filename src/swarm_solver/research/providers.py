import asyncio
import html
import logging
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx

from ..config import Settings
from ..logs import LOG
from .models import Origin

log = logging.getLogger(f"{LOG}.research.providers")

HN_API = "https://hn.algolia.com/api/v1/search"


@dataclass
class Raw:
    title: str
    url: str
    snippet: str = ""
    content: str | None = None  # texte complet déjà fourni par la source


class SearchProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[Raw]: ...


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


class DdgsSearch:
    """DuckDuckGo via le paquet `ddgs` : aucune clé, aucune infrastructure."""

    def __init__(self, region: str):
        self._region = region

    async def search(self, query: str, limit: int) -> list[Raw]:
        from ddgs import DDGS

        rows = await asyncio.to_thread(lambda: DDGS().text(query, region=self._region, max_results=limit))
        return [Raw(r["title"], r["href"], r.get("body", "")) for r in rows]


class SearxngSearch:
    """Instance SearXNG auto-hébergée (le format JSON doit être activé dans son settings.yml)."""

    def __init__(self, base_url: str, region: str, client: httpx.AsyncClient):
        self._base, self._lang, self._client = base_url.rstrip("/"), region.split("-")[0], client

    async def search(self, query: str, limit: int) -> list[Raw]:
        resp = await self._client.get(
            f"{self._base}/search", params={"q": query, "format": "json", "language": self._lang}
        )
        resp.raise_for_status()
        rows = resp.json().get("results", [])[:limit]
        return [Raw(r.get("title", ""), r["url"], r.get("content", "")) for r in rows if r.get("url")]


class HackerNewsSearch:
    """API publique Algolia de Hacker News : histoires et commentaires, texte complet inclus."""

    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def search(self, query: str, limit: int) -> list[Raw]:
        resp = await self._client.get(HN_API, params={"query": query, "tags": "(story,comment)", "hitsPerPage": limit})
        resp.raise_for_status()
        out = []
        for h in resp.json().get("hits", []):
            text = _strip_html(h.get("comment_text") or h.get("story_text") or "")
            item_url = f"https://news.ycombinator.com/item?id={h['objectID']}"
            title = h.get("story_title") or h.get("title") or "Hacker News"
            out.append(Raw(title, item_url, text[:300], content=text or None))
        return out


class RedditViaWeb:
    """Reddit refuse les accès anonymes (403) : on passe par une recherche `site:reddit.com` et on garde l'extrait."""

    def __init__(self, web: SearchProvider):
        self._web = web

    async def search(self, query: str, limit: int) -> list[Raw]:
        rows = await self._web.search(f"site:reddit.com {query}", limit * 2)
        kept = [r for r in rows if (urlparse(r.url).hostname or "").endswith("reddit.com")]
        return [Raw(r.title, r.url, r.snippet, content=r.snippet or None) for r in kept[:limit]]


def build_providers(settings: Settings, client: httpx.AsyncClient) -> dict[Origin, SearchProvider]:
    if settings.search_provider == "searxng":
        if not settings.searxng_url:
            raise ValueError("SWARM_SEARCH_PROVIDER=searxng demande SWARM_SEARXNG_URL")
        web: SearchProvider = SearxngSearch(settings.searxng_url, settings.search_region, client)
    elif settings.search_provider == "ddgs":
        web = DdgsSearch(settings.search_region)
    else:
        raise ValueError(f"Fournisseur de recherche inconnu: {settings.search_provider!r} (ddgs ou searxng)")
    return {"web": web, "reddit": RedditViaWeb(web), "hn": HackerNewsSearch(client)}
