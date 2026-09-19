import asyncio
import ipaddress
import logging
import socket
from collections import defaultdict
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from ..config import Settings
from ..logs import LOG

log = logging.getLogger(f"{LOG}.research.fetch")

MAX_BYTES = 1_500_000
MAX_CHARS = 30_000
MAX_REDIRECTS = 3
TEXT_TYPES = ("text/html", "application/xhtml", "text/plain")
TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"}

Resolver = Callable[[str], Awaitable[list[str]]]


class UnsafeURL(Exception):
    pass


def normalize_url(url: str) -> str:
    """Forme canonique pour dédupliquer : sans fragment, sans paramètres de suivi, hôte en minuscules."""
    p = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(p.query) if not k.startswith("utm_") and k not in TRACKING_PARAMS]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", urlencode(query), ""))


async def _resolve(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [i[4][0] for i in infos]


async def assert_public(url: str, resolve: Resolver = _resolve) -> None:
    """Refuse tout ce qui n'est pas du http(s) vers une adresse publique (localhost, réseau local, métadonnées cloud)."""
    p = urlparse(url)
    if p.scheme not in {"http", "https"} or not p.hostname:
        raise UnsafeURL(f"schéma ou hôte refusé: {url}")
    try:
        addresses = [p.hostname] if _is_ip(p.hostname) else await resolve(p.hostname)
    except OSError as e:
        raise UnsafeURL(f"résolution impossible pour {p.hostname}: {e}") from e
    for addr in addresses:
        if not ipaddress.ip_address(addr).is_global:
            raise UnsafeURL(f"adresse non publique refusée: {p.hostname} -> {addr}")


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


class Fetcher:
    """Télécharge des pages publiques et en extrait le texte principal.

    Garde-fous : adresses publiques uniquement (redirections comprises), robots.txt respecté, taille et types
    de contenu bornés, au plus 2 requêtes simultanées par domaine.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient, resolve: Resolver = _resolve):
        self._client, self._resolve = client, resolve
        self._ua = settings.user_agent
        self._sem = asyncio.Semaphore(settings.fetch_concurrency)
        self._domain_sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(2))
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def _get(self, url: str) -> tuple[str, str] | None:
        """GET avec redirections suivies à la main pour re-vérifier chaque étape. Retourne (contenu, type)."""
        for _ in range(MAX_REDIRECTS + 1):
            await assert_public(url, self._resolve)
            async with self._client.stream("GET", url, headers={"User-Agent": self._ua}, follow_redirects=False) as resp:
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                if resp.status_code != 200:
                    log.debug("fetch %s -> HTTP %d", url, resp.status_code)
                    return None
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if not ctype.startswith(TEXT_TYPES):
                    log.debug("fetch %s -> type ignoré: %s", url, ctype)
                    return None
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        break
                return bytes(body).decode(resp.encoding or "utf-8", errors="replace"), ctype
        log.debug("fetch %s -> trop de redirections", url)
        return None

    async def _allowed(self, url: str) -> bool:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        async with self._robots_locks[origin]:
            if origin not in self._robots:
                parser: RobotFileParser | None = None
                try:
                    got = await self._get(f"{origin}/robots.txt")
                    if got:
                        parser = RobotFileParser()
                        parser.parse(got[0].splitlines())
                except (httpx.HTTPError, UnsafeURL) as e:
                    log.debug("robots.txt illisible pour %s: %s", origin, e)
                self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(self._ua, url)

    async def get_text(self, url: str) -> str | None:
        """Texte principal de la page, ou None (refusé, illisible, pas du texte, robots.txt)."""
        host = urlparse(url).netloc
        async with self._sem, self._domain_sems[host]:
            try:
                if not await self._allowed(url):
                    log.debug("fetch %s -> interdit par robots.txt", url)
                    return None
                got = await self._get(url)
            except (httpx.HTTPError, UnsafeURL) as e:
                log.debug("fetch %s -> %s: %s", url, type(e).__name__, e)
                return None
        if not got:
            return None
        content, ctype = got
        if ctype.startswith("text/plain"):
            return content[:MAX_CHARS]
        text = await asyncio.to_thread(
            trafilatura.extract, content, include_comments=False, include_tables=False, favor_recall=True
        )
        return (text or "")[:MAX_CHARS] or None
