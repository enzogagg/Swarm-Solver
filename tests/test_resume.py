import argparse
import io
import json
from dataclasses import replace

import pytest
from rich.console import Console

from swarm_solver import cli
from swarm_solver.config import Settings
from swarm_solver.pipeline import SolveParams, solve
from swarm_solver.research import Research, run_research
from swarm_solver.research.models import Finding, Query, ResearchPlan, Source, SourcedFinding
from swarm_solver.research.providers import Raw

from test_pipeline import SPEC, FakeLLM
from test_research import ResearchLLM, StubFetcher, StubProvider

PARAMS = SolveParams(ideas_target=60, per_call=5, pool=5, top=2)


class NoCallsLLM(FakeLLM):
    """Reprendre un run complet ne doit consulter aucun modèle."""

    async def structured(self, *a, **k):
        raise AssertionError("appel LLM inattendu (structured)")

    async def text(self, *a, **k):
        raise AssertionError("appel LLM inattendu (text)")

    async def embed(self, *a, **k):
        raise AssertionError("appel LLM inattendu (embed)")


def tags(llm: FakeLLM) -> list[str]:
    return [t for t, _ in llm.prompts]


async def first_run(tmp_path):
    settings = replace(Settings(), runs_dir=tmp_path, web_enabled=False)
    llm = FakeLLM(buckets=1000)
    result = await solve(llm, settings, Console(quiet=True), SPEC, PARAMS)
    return settings, result


# --- reprise ------------------------------------------------------------------------------------------------


async def test_resuming_a_complete_run_reloads_everything_without_any_model_call(tmp_path):
    settings, first = await first_run(tmp_path)
    assert first.reserve and first.dropped is not None

    again = await solve(NoCallsLLM(), settings, Console(quiet=True), SPEC, PARAMS, resume_dir=first.run_dir)

    assert [r.idea.id for r in again.survivors] == [r.idea.id for r in first.survivors]
    assert again.stats == first.stats
    assert sorted(i.id for i, _ in again.reserve) == sorted(i.id for i, _ in first.reserve)
    assert {i.id: v.score for i, v in again.reserve} == {i.id: v.score for i, v in first.reserve}  # notes conservées
    assert [i.id for i, _ in again.dropped] == [i.id for i, _ in first.dropped]
    assert again.run_dir == first.run_dir


async def test_resuming_a_run_stopped_before_the_audit_only_redoes_the_missing_stages(tmp_path):
    settings, first = await first_run(tmp_path)
    for name in ("audits.jsonl", "dropped.jsonl", "reserve.jsonl", "report.md", "stats.json"):
        (first.run_dir / name).unlink()

    llm = FakeLLM(buckets=1000)
    again = await solve(llm, settings, Console(quiet=True), SPEC, PARAMS, resume_dir=first.run_dir)

    used = " ".join(tags(llm))
    assert "essaim" not in used and "analyse" not in used  # déjà sur disque: pas refaits
    assert "filtre" in used and "audit" in used and "rapport" in used
    assert (again.run_dir / "report.md").exists()
    assert again.unique_count == first.unique_count


async def test_a_run_from_an_older_version_without_reserve_file_rebuilds_its_reserve(tmp_path):
    settings, first = await first_run(tmp_path)
    (first.run_dir / "reserve.jsonl").unlink()

    again = await solve(NoCallsLLM(), settings, Console(quiet=True), SPEC, PARAMS, resume_dir=first.run_dir)

    assert sorted(i.id for i, _ in again.reserve) == sorted(i.id for i, _ in first.reserve)
    assert {v.score for _, v in again.reserve} == {5}  # la note d'origine est perdue: note neutre


async def test_an_analysis_in_an_old_format_is_recomputed(tmp_path):
    settings, first = await first_run(tmp_path)
    path = first.run_dir / "analysis.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    del old["adversarial"]  # format d'avant ce champ obligatoire
    path.write_text(json.dumps(old), encoding="utf-8")

    llm = FakeLLM(buckets=1000)
    await solve(llm, settings, Console(quiet=True), SPEC, PARAMS, resume_dir=first.run_dir)
    assert any(t.startswith("analyse") for t in tags(llm))


async def test_research_stopped_before_the_synthesis_gets_its_brief_written_on_resume(tmp_path):
    settings, first = await first_run(tmp_path)
    for name in ("analysis.json", "ideas_raw.jsonl", "ideas_unique.jsonl", "audits.jsonl", "dropped.jsonl",
                 "reserve.jsonl", "report.md"):
        (first.run_dir / name).unlink()
    research = Research(
        plan=ResearchPlan(queries=[Query(text="requête test", purpose="p")]),
        sources=[Source(id=1, url="https://good.example/a", title="Guide", origin="web", query="requête test", chars=99)],
        findings=[SourcedFinding(claim="Le fait utile est confirmé", quote="citation de la page", kind="law",
                                 relevance=8, source_id=1)],
    )  # dossier vide: l'arrêt a eu lieu entre l'extraction et la synthèse
    (first.run_dir / "research.json").write_text(research.model_dump_json(), encoding="utf-8")

    llm = ResearchLLM()
    again = await solve(llm, settings, Console(quiet=True), SPEC, PARAMS, resume_dir=first.run_dir)

    assert not again.brief.empty  # il était vide sur disque: la synthèse a donc bien été refaite
    assert any("WEB RESEARCH BRIEF" in p for t, p in llm.prompts if t.startswith("essaim"))
    saved = Research.model_validate_json((first.run_dir / "research.json").read_text(encoding="utf-8"))
    assert not saved.brief.empty  # le dossier est bien réécrit sur disque


# --- la recherche est écrite dès que les faits sont extraits ----------------------------------------------------


async def test_research_is_reported_after_extraction_and_again_after_the_synthesis():
    snapshots = []
    web = StubProvider([Raw("Guide", "https://good.example/guide", "s")])
    await run_research(
        ResearchLLM(), Settings(), Console(quiet=True), SPEC,
        providers={"web": web, "hn": web, "reddit": web},
        fetcher=StubFetcher({"https://good.example/guide": "page utile " * 20}),
        on_progress=lambda r: snapshots.append((len(r.findings), r.brief.empty)),
    )
    # 1) faits extraits, dossier pas encore rédigé: c'est ce qui manquait quand un run s'arrêtait à ce moment-là
    assert snapshots[0][0] > 0 and snapshots[0][1] is True
    assert snapshots[-1][1] is False


# --- trouver le run ----------------------------------------------------------------------------------------------


def make_run(root, name):
    (root / name).mkdir(parents=True)
    (root / name / "problem.json").write_text(SPEC.model_dump_json(), encoding="utf-8")
    return root / name


def test_find_run_by_latest_name_and_path(tmp_path):
    old, new = make_run(tmp_path, "20260101-100000"), make_run(tmp_path, "20260202-100000")
    (tmp_path / "20260303-100000").mkdir()  # dossier sans problem.json: ignoré
    assert cli.find_run(None, tmp_path) == new
    assert cli.find_run("latest", tmp_path) == new
    assert cli.find_run("20260101-100000", tmp_path) == old
    assert cli.find_run(str(old), tmp_path) == old


def test_find_run_reports_a_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.find_run("nope", tmp_path)
    with pytest.raises(FileNotFoundError):
        cli.find_run(None, tmp_path)


async def test_resume_command_prints_an_error_and_returns_1_for_an_unknown_run(tmp_path, monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, width=150))
    args = argparse.Namespace(run="introuvable", ideas=10, per_call=5, pool=3, top=2, no_followup=True)
    code = await cli.cmd_resume(args, replace(Settings(), runs_dir=tmp_path))
    assert code == 1 and "Run introuvable" in out.getvalue()


def test_audits_saved_with_the_old_legality_lens_name_are_still_readable():
    """Les premiers runs (avant le renommage legal_ethics -> legality) doivent pouvoir être repris."""
    from swarm_solver.schemas import DeepResult, Idea, LensVerdict

    lens = LensVerdict(score=8, blocking=False).model_dump()
    old = {"idea": Idea(id="I1", lens="l", title="t", description="d", steps=["a", "b"]).model_dump(), "gate_score": 7,
           "lenses": {"feasibility": lens, "constraints_risk": lens, "legal_ethics": lens}}
    result = DeepResult.model_validate(old)
    assert "legality" in result.lenses and "legal_ethics" not in result.lenses
    assert result.rejection is None
