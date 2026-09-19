import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    host: str = "http://localhost:11434"
    # Petit modèle rapide : essaim d'idées + filtre éliminatoire (des milliers d'appels).
    fast_model: str = "qwen3:8b"
    # Modèle plus solide : interview, analyse, vérification approfondie, rapport final.
    strong_model: str = "qwen3:14b"
    embed_model: str = "nomic-embed-text"
    # Requêtes simultanées côté client. À aligner sur OLLAMA_NUM_PARALLEL côté serveur.
    concurrency: int = 4
    num_ctx: int = 8192
    language: str = "français"
    # Droit de référence pour juger la légalité des moyens.
    jurisdiction: str = "France"
    dedup_threshold: float = 0.88
    runs_dir: Path = Path("runs")
    # Recherche web (les requêtes quittent la machine ; les modèles restent locaux).
    web_enabled: bool = True
    search_provider: str = "ddgs"  # "ddgs" (sans configuration) ou "searxng" (instance auto-hébergée)
    searxng_url: str = ""
    search_region: str = "fr-fr"
    research_queries: int = 10
    results_per_query: int = 6
    max_pages: int = 24
    fetch_concurrency: int = 6
    fetch_timeout: float = 15.0
    user_agent: str = "swarm-solver/0.1 (personal research tool)"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            host=os.getenv("OLLAMA_HOST", d.host),
            fast_model=os.getenv("SWARM_FAST_MODEL", d.fast_model),
            strong_model=os.getenv("SWARM_STRONG_MODEL", d.strong_model),
            embed_model=os.getenv("SWARM_EMBED_MODEL", d.embed_model),
            concurrency=int(os.getenv("SWARM_CONCURRENCY", d.concurrency)),
            num_ctx=int(os.getenv("SWARM_NUM_CTX", d.num_ctx)),
            language=os.getenv("SWARM_LANGUAGE", d.language),
            jurisdiction=os.getenv("SWARM_JURISDICTION", d.jurisdiction),
            dedup_threshold=float(os.getenv("SWARM_DEDUP_THRESHOLD", d.dedup_threshold)),
            runs_dir=Path(os.getenv("SWARM_RUNS_DIR", d.runs_dir)),
            web_enabled=os.getenv("SWARM_WEB", "1").lower() not in {"0", "false", "no", "non"},
            search_provider=os.getenv("SWARM_SEARCH_PROVIDER", d.search_provider),
            searxng_url=os.getenv("SWARM_SEARXNG_URL", d.searxng_url),
            search_region=os.getenv("SWARM_SEARCH_REGION", d.search_region),
            research_queries=int(os.getenv("SWARM_RESEARCH_QUERIES", d.research_queries)),
            results_per_query=int(os.getenv("SWARM_RESULTS_PER_QUERY", d.results_per_query)),
            max_pages=int(os.getenv("SWARM_MAX_PAGES", d.max_pages)),
        )
