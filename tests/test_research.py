import io
from dataclasses import replace

import httpx
import pytest
from rich.console import Console

from swarm_solver.config import Settings
from swarm_solver.pipeline import SolveParams, solve
from swarm_solver.research import Research, ResearchBrief, run_research
from swarm_solver.research.agent import select_hits
from swarm_solver.research.fetch import Fetcher, UnsafeURL, assert_public, normalize_url
from swarm_solver.research.models import (
    BriefDraft,
    BriefPoint,
    DraftPoint,
    Finding,
    Hit,
    PageFindings,
    Query,
    ResearchPlan,
    Source,
    SourcedFinding,
)
from swarm_solver.research.providers import HackerNewsSearch, Raw, RedditViaWeb, SearxngSearch
from swarm_solver.session import Session

from test_pipeline import SPEC, FakeLLM

PUBLIC, PRIVATE = "93.184.216.34", "10.0.0.5"
HOSTS = {"good.example": PUBLIC, "other.example": PUBLIC, "internal.example": PRIVATE}


async def fake_resolve(host: str) -> list[str]:
    return [HOSTS[host]]


ARTICLE = "<html><body><article><h1>Procédure</h1>" + "".join(
    f"<p>Étape {i}: la salariée doit saisir par écrit le service des ressources humaines et conserver une copie "
    f"datée de chaque échange, puis demander un entretien formel avec un représentant du personnel.</p>"
    for i in range(1, 8)
) + "</article></body></html>"


def make_fetcher(handler) -> Fetcher:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Fetcher(replace(Settings(), fetch_concurrency=2), client, resolve=fake_resolve)


def html_response(body: str = ARTICLE, ctype: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": ctype}, content=body.encode())


# --- garde-fous réseau ---------------------------------------------------------------------------------------


def test_normalize_url_drops_tracking_and_fragment():
    assert normalize_url("HTTPS://Good.Example/a/?utm_source=x&id=3&fbclid=y#top") == "https://good.example/a?id=3"


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd",
                                 "http://internal.example/x", "ftp://good.example/"])
async def test_assert_public_refuses_local_and_odd_targets(url):
    with pytest.raises(UnsafeURL):
        await assert_public(url, fake_resolve)


async def test_assert_public_accepts_public_host():
    await assert_public("https://good.example/page", fake_resolve)


async def test_fetcher_extracts_main_text():
    fetcher = make_fetcher(lambda req: httpx.Response(404) if req.url.path == "/robots.txt" else html_response())
    text = await fetcher.get_text("https://good.example/guide")
    assert text and "ressources humaines" in text


async def test_fetcher_respects_robots_txt():
    hits = []

    def handler(req):
        hits.append(req.url.path)
        if req.url.path == "/robots.txt":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"User-agent: *\nDisallow: /private")
        return html_response()

    fetcher = make_fetcher(handler)
    assert await fetcher.get_text("https://good.example/private/doc") is None
    assert "/private/doc" not in hits
    assert await fetcher.get_text("https://good.example/public/doc")


async def test_fetcher_refuses_redirect_to_private_address():
    hits = []

    def handler(req):
        hits.append(req.url.host)
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "http://internal.example/secret"})

    fetcher = make_fetcher(handler)
    assert await fetcher.get_text("https://good.example/start") is None
    assert "internal.example" not in hits


async def test_fetcher_ignores_non_text_content():
    fetcher = make_fetcher(lambda req: httpx.Response(404) if req.url.path == "/robots.txt"
                           else html_response("%PDF-1.4", "application/pdf"))
    assert await fetcher.get_text("https://good.example/file.pdf") is None


# --- fournisseurs --------------------------------------------------------------------------------------------


async def test_hacker_news_returns_comment_text_without_html():
    payload = {"hits": [{"objectID": "42", "comment_text": "<p>Documenter chaque incident &amp; prévenir les RH</p>",
                         "story_title": "Ask HN: toxic coworker"}]}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
    [raw] = await HackerNewsSearch(client).search("toxic coworker", 5)
    assert raw.content == "Documenter chaque incident & prévenir les RH"
    assert raw.url == "https://news.ycombinator.com/item?id=42"


async def test_searxng_parses_json_results():
    payload = {"results": [{"title": "T", "url": "https://good.example/a", "content": "c"}, {"title": "sans url"}]}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
    rows = await SearxngSearch("http://searx.local:8080/", "fr-fr", client).search("q", 5)
    assert [r.url for r in rows] == ["https://good.example/a"]


async def test_reddit_via_web_keeps_only_reddit_and_uses_snippet():
    class Web:
        async def search(self, query, limit):
            assert query.startswith("site:reddit.com ")
            return [Raw("post", "https://www.reddit.com/r/x/comments/1/a", "mon expérience"),
                    Raw("autre", "https://blog.example/b", "hors sujet")]

    [raw] = await RedditViaWeb(Web()).search("collègue", 5)
    assert raw.url.startswith("https://www.reddit.com") and raw.content == "mon expérience"


# --- sélection, citations ------------------------------------------------------------------------------------


def _hit(url: str, query: str) -> Hit:
    return Hit(title=url, url=url, origin="web", query=query)


def test_select_hits_round_robin_dedupes_and_caps_per_domain():
    hits = [_hit("https://a.example/1?utm_source=x", "q1"), _hit("https://a.example/2", "q1"),
            _hit("https://a.example/3", "q1"), _hit("https://a.example/4", "q1"),
            _hit("https://a.example/1", "q2"), _hit("https://b.example/1", "q2")]
    chosen = select_hits(hits, max_pages=10, per_domain=2)
    # Tour 1: a/1 (q1), doublon normalisé écarté (q2). Tour 2: a/2 (q1), b/1 (q2). Ensuite a.example est plafonné à 2.
    assert [h.url for h in chosen] == ["https://a.example/1?utm_source=x", "https://a.example/2", "https://b.example/1"]


def test_brief_build_drops_invented_source_ids_and_unsupported_points():
    draft = BriefDraft(
        points=[DraftPoint(text="vrai", sources=[1, 99], section="law_and_procedure"),
                DraftPoint(text="inventé", sources=[99], section="law_and_procedure"),
                DraftPoint(text="échec", sources=[2], section="what_backfired")],
        unknowns=["?"],
    )
    brief = ResearchBrief.build(draft, known_ids={1, 2})
    assert [p.text for p in brief.what_backfired] == ["échec"]
    assert [(p.text, p.sources) for p in brief.law_and_procedure] == [("vrai", [1])]
    assert "[S1]" in brief.to_prompt() and "[S99]" not in brief.to_prompt()


# --- recherche de bout en bout (faux LLM, faux fournisseurs) --------------------------------------------------


class StubProvider:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def search(self, query, limit):
        self.queries.append(query)
        return self.rows


class StubFetcher:
    def __init__(self, pages):
        self.pages, self.fetched = pages, []

    async def get_text(self, url):
        self.fetched.append(url)
        return self.pages.get(url)


class ResearchLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.extraction_prompts: list[str] = []

    async def structured(self, model, system, user, schema, **kw):
        if schema is ResearchPlan:
            return ResearchPlan(queries=[Query(text="procédure salarié conflit hiérarchie", purpose="droit", sources=["web", "hn"])])
        if schema is PageFindings:
            self.extraction_prompts.append(user)
            page = user.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
            if "utile" not in page:
                return PageFindings(findings=[])
            return PageFindings(findings=[
                Finding(claim="fait utile", quote=page[:30], kind="procedure", relevance=8),
                Finding(claim="anecdote", quote=page[:30], kind="case", relevance=2),
                # Le modèle invente un fait bien noté: la citation n'existe pas dans la page.
                Finding(claim="fait inventé", quote="citation totalement inventée par le modèle", kind="law", relevance=9),
            ])
        if schema is BriefDraft:
            return BriefDraft(points=[DraftPoint(text="Le fait utile est confirmé", sources=[1, 99], section="law_and_procedure")])
        return await super().structured(model, system, user, schema, **kw)


async def test_run_research_builds_sourced_brief_and_skips_fetch_for_inline_content():
    web = StubProvider([Raw("Guide", "https://good.example/guide", "s"), Raw("Vide", "https://other.example/vide", "s")])
    hn = StubProvider([Raw("HN", "https://news.ycombinator.com/item?id=1", "s", content="témoignage utile " * 10)])
    fetcher = StubFetcher({"https://good.example/guide": "page utile " * 20})
    research = await run_research(ResearchLLM(), Settings(), Console(quiet=True), SPEC,
                                  providers={"web": web, "hn": hn, "reddit": web}, fetcher=fetcher)

    assert research is not None
    assert "https://news.ycombinator.com/item?id=1" not in fetcher.fetched  # contenu déjà fourni par l'API
    assert {s.url for s in research.sources} == {"https://good.example/guide", "https://news.ycombinator.com/item?id=1"}
    assert all(f.relevance >= 4 for f in research.findings)  # l'anecdote (2/10) est filtrée
    assert [f.claim for f in research.findings] == ["fait utile", "fait utile"]  # le fait à citation inventée est rejeté
    assert [(p.text, p.sources) for p in research.brief.law_and_procedure] == [("Le fait utile est confirmé", [1])]


async def test_run_research_returns_none_when_nothing_useful():
    web = StubProvider([Raw("Vide", "https://other.example/vide", "s")])
    fetcher = StubFetcher({"https://other.example/vide": "contenu sans intérêt " * 10})
    assert await run_research(ResearchLLM(), Settings(), Console(quiet=True), SPEC,
                              providers={"web": web, "hn": web, "reddit": web}, fetcher=fetcher) is None


# --- intégration au pipeline et à la session ------------------------------------------------------------------


def _canned_research() -> Research:
    return Research(
        plan=ResearchPlan(queries=[Query(text="requête test", purpose="p")]),
        sources=[Source(id=1, url="https://good.example/guide", title="Guide CSE", origin="web", query="requête test", chars=100)],
        findings=[SourcedFinding(claim="c", quote="citation de la page", kind="procedure", relevance=8, source_id=1)],
        brief=ResearchBrief(law_and_procedure=[BriefPoint(text="Saisir le CSE", sources=[1])]),
    )


async def test_pipeline_feeds_brief_to_analysis_swarm_and_audits_and_appends_sources(tmp_path, monkeypatch):
    async def fake_run_research(*a, **k):
        return _canned_research()

    monkeypatch.setattr("swarm_solver.pipeline.run_research", fake_run_research)

    class CitingLLM(FakeLLM):
        async def text(self, model, system, user, *, temperature=0.4, tag=""):
            await super().text(model, system, user, temperature=temperature, tag=tag)
            return "Recommandation appuyée par le CSE [S1]." if tag == "rapport" else "x"

    llm = CitingLLM()
    settings = replace(Settings(), runs_dir=tmp_path)
    result = await solve(llm, settings, Console(quiet=True), SPEC, SolveParams(ideas_target=20, per_call=5, pool=5, top=2))

    def saw_brief(prefix):
        return any("WEB RESEARCH BRIEF" in p for tag, p in llm.prompts if tag.startswith(prefix))

    assert saw_brief("analyse") and saw_brief("essaim") and saw_brief("audit legality") and saw_brief("rapport")
    assert not saw_brief("filtre"), "le filtre rapide tourne sur des centaines d'idées : pas de dossier dans ce prompt"
    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Sources" in report and "https://good.example/guide" in report
    assert (result.run_dir / "research.json").exists()


async def test_pipeline_continues_when_web_research_fails(tmp_path, monkeypatch):
    async def broken(*a, **k):
        raise RuntimeError("réseau coupé")

    monkeypatch.setattr("swarm_solver.pipeline.run_research", broken)
    out = io.StringIO()
    settings = replace(Settings(), runs_dir=tmp_path)
    result = await solve(FakeLLM(), settings, Console(file=out, width=200), SPEC,
                         SolveParams(ideas_target=10, per_call=5, pool=3, top=2))

    assert result.research is None and (result.run_dir / "report.md").exists()
    assert "Recherche web indisponible" in out.getvalue()


async def test_session_web_and_sources_commands(tmp_path, monkeypatch):
    research = _canned_research()

    async def fake_ask_web(llm, settings, console, spec, current, question):
        assert question == "délai pour saisir le CSE ?"
        return "Sous 5 jours [S1].", research

    monkeypatch.setattr("swarm_solver.session.ask_web", fake_ask_web)
    settings = replace(Settings(), runs_dir=tmp_path, web_enabled=False)
    llm, params = FakeLLM(), SolveParams(ideas_target=10, per_call=5, pool=3, top=2)
    result = await solve(llm, settings, Console(quiet=True), SPEC, params)

    script = iter(["recherche délai pour saisir le CSE ?", "sources", "q"])

    async def read(_):
        return next(script)

    # Web désactivé: la commande doit refuser sans appeler ask_web.
    console = Console(file=io.StringIO(), record=True, width=200)
    await Session(llm, settings, console, result, params, read).run()
    assert "désactivée" in console.export_text()

    # Web activé: la réponse et les sources citées s'affichent, le dossier est fusionné dans le run.
    script = iter(["recherche délai pour saisir le CSE ?", "sources", "q"])
    console = Console(file=io.StringIO(), record=True, width=200)
    await Session(llm, replace(settings, web_enabled=True), console, result, params, read).run()
    out = console.export_text()
    assert "Sous 5 jours" in out and "Guide CSE" in out and "[S1]" in out
    assert result.research is research and (result.run_dir / "research.json").exists()


async def test_extraction_prompt_never_contains_the_users_own_facts():
    """Un petit modèle recopie la fiche comme si elle venait de la page: l'extracteur ne doit donc pas la voir."""
    spec = SPEC.model_copy(update={"key_facts": ["fait confidentiel de l'utilisateur"], "context": "contexte privé"})
    web = StubProvider([Raw("Guide", "https://good.example/guide", "s")])
    llm = ResearchLLM()
    await run_research(llm, Settings(), Console(quiet=True), spec, providers={"web": web, "hn": web, "reddit": web},
                       fetcher=StubFetcher({"https://good.example/guide": "page utile " * 20}))
    assert llm.extraction_prompts
    assert all("fait confidentiel" not in p and "contexte privé" not in p for p in llm.extraction_prompts)


@pytest.mark.parametrize("quote, expected", [
    ("La salariée doit saisir le CSE", True),
    ("la   salariée doit\nsaisir le CSE", True),  # casse et espaces ignorés
    ("L\u2019employeur doit agir", True),  # apostrophe typographique
    ("La salariée ... saisir le CSE dans un délai", True),  # ellipse: chaque segment doit exister
    ("Le salarié peut démissionner librement", False),
    ("« saisir le CSE, dans un délai de huit jours »", True),  # guillemets, virgule: ignorés
    ("saisir le CSE dans un délai de dix jours", False),  # un mot différent suffit à rejeter
    ("saisir le CSE dans un dél", True),  # mot coupé en fin de citation: toléré
    ("saisir le SCE dans un délai", False),  # mot interne différent: rejeté
    ("...", False),
])
def test_quote_in_page(quote, expected):
    from swarm_solver.research.agent import quote_in_page

    page = "Règle: La salariée doit saisir le CSE dans un délai de huit jours. L'employeur doit agir."
    assert quote_in_page(quote, page) is expected


def _finding(source_id: int, claim: str) -> SourcedFinding:
    return SourcedFinding(source_id=source_id, claim=claim, quote="citation de la page", kind="law", relevance=8)


def test_brief_citations_must_be_supported_by_a_finding_of_the_cited_source():
    findings = [
        _finding(1, "Le salarié n'est pas obligé de se présenter à l'entretien préalable"),
        _finding(3, "Une étude indique que 85 % des employés sont confrontés à un conflit chaque année"),
    ]
    draft = BriefDraft(points=[
        # Bonne phrase, mauvaise source (S3): le code la rattache à S1, dont un fait la soutient.
        DraftPoint(text="Le salarié n'est pas obligé de se présenter à l'entretien préalable", sources=[3], section="law_and_procedure"),
        # Phrase qu'aucun fait ne soutient, même avec une source valide: supprimée.
        DraftPoint(text="Le CSE doit obligatoirement rendre un avis sous trois jours", sources=[1], section="law_and_procedure"),
        # Bonne source: conservée telle quelle.
        DraftPoint(text="85 % des employés sont confrontés à un conflit chaque année", sources=[3], section="tactics_seen_in_practice"),
    ])
    brief = ResearchBrief.build(draft, known_ids={1, 3}, findings=findings)
    assert [(p.text[:20], p.sources) for p in brief.law_and_procedure] == [("Le salarié n'est pas", [1])]
    assert [p.sources for p in brief.tactics_seen_in_practice] == [[3]]
