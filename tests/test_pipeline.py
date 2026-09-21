import io
import itertools
import json
import logging
import zlib
from dataclasses import replace

import numpy as np
from rich.console import Console

from swarm_solver.config import Settings
from swarm_solver.dedup import dedupe_indices
from swarm_solver.interview import run_interview
from swarm_solver.logs import TRACE, setup_logging
from swarm_solver.pipeline import RunResult, SolveParams, ranked_for_report, solve
from swarm_solver.schemas import (
    BASELINE_RED_LINES,
    Actor,
    Analysis,
    DeepResult,
    GateVerdict,
    Idea,
    IdeaBatch,
    IdeaDraft,
    InterviewStep,
    LensVerdict,
    ProblemSpec,
)
from swarm_solver.session import Session

ANALYSIS = Analysis(
    adversarial=True, root_causes=["cause"], actors=[Actor(name="A", incentives="i", leverage="l")], leverage_points=["levier"]
)
SPEC = ProblemSpec(title="T", stated_request="r", underlying_goal="g", context="c", red_lines=["pas de médias"])


class FakeLLM:
    """Remplace Ollama : sorties déterministes. `buckets` borne le nombre de vecteurs d'embedding distincts."""

    def __init__(self, buckets: int = 20):
        self._n = itertools.count()
        self.buckets = buckets
        self.interview_calls = 0
        self.prompts: list[tuple[str, str]] = []

    async def structured(self, model, system, user, schema, *, temperature=0.3, seed=None, tag=""):
        self.prompts.append((tag, system + "\n" + user))
        if schema is InterviewStep:
            self.interview_calls += 1
            return InterviewStep(ready=self.interview_calls > 1, questions=["Quel budget ?"])
        if schema is ProblemSpec:
            return SPEC
        if schema is Analysis:
            return ANALYSIS
        if schema is IdeaBatch:
            kind = "variant" if "BASE SOLUTION" in user else "idea"
            return IdeaBatch(ideas=[IdeaDraft(title=f"{kind} {next(self._n)}", description="d", steps=["s1", "s2"]) for _ in range(5)])
        if schema is GateVerdict:
            bad = "idea 3\n" in user
            return GateVerdict(violates_red_line=False, illegal_or_harmful=bad, plausible=True, score=7, reason="ok")
        if schema is LensVerdict:
            return LensVerdict(score=8, blocking=False)
        raise AssertionError(schema)

    async def text(self, model, system, user, *, temperature=0.4, tag=""):
        self.prompts.append((tag, system + "\n" + user))
        return {"rapport": "corps du rapport", "conseiller": "réponse du conseiller"}.get(tag, "plan détaillé")

    async def embed(self, model, texts):
        out = np.zeros((len(texts), self.buckets), dtype=np.float32)
        for row, t in enumerate(texts):
            out[row, zlib.crc32(t.encode()) % self.buckets] = 1.0
        return out


def _idea(n: int) -> Idea:
    return Idea(id=f"I{n}", lens="l", title=f"idée {n}", description="d", steps=["s1", "s2"])


def _deep(n: int, *, feasibility=(8, False), risk=(8, False), legality=(8, False), gate=7) -> DeepResult:
    def lens(v):
        return LensVerdict(score=v[0], blocking=v[1], issues=["x"])

    return DeepResult(
        idea=_idea(n), gate_score=gate,
        lenses={"feasibility": lens(feasibility), "constraints_risk": lens(risk), "legality": lens(legality)},
    )


# --- dédup ---------------------------------------------------------------------------------------------------


def test_dedupe_indices_drops_near_duplicates():
    v = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    assert dedupe_indices(v, 0.9) == [0, 2]


def test_dedupe_indices_empty():
    assert dedupe_indices(np.empty((0, 4), dtype=np.float32), 0.9) == []


def test_dedupe_indices_fixed_references_are_compared_but_not_returned():
    v = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    assert dedupe_indices(v, 0.9, fixed=1) == [2]


# --- règles de rejet et garde-fous ---------------------------------------------------------------------------


def test_risk_alone_never_rejects():
    r = _deep(1, risk=(1, False))
    assert r.rejection is None
    assert r.risk_level == "high"


def test_only_an_established_illegal_or_harmful_verdict_rejects_by_default():
    assert _deep(1, legality=(8, True)).rejection.startswith("illégal")
    assert _deep(2, legality=(4, False)).rejection is None  # note basse sans verdict bloquant: alerte, pas rejet


def test_difficulty_and_doubt_raise_warnings_but_never_remove_an_idea():
    """L'utilisateur décide: infaisable, risqué ou de légalité douteuse restent dans le classement, avec une alerte."""
    assert _deep(1, feasibility=(2, True)).rejection is None
    assert _deep(2, feasibility=(5, True)).warnings == ["faisabilité"]  # bloquant selon l'auditeur, même noté 5
    assert _deep(3, feasibility=(3, False)).warnings == ["faisabilité"]
    assert _deep(4, legality=(4, False)).warnings == ["légalité"]
    assert _deep(5, risk=(2, False)).warnings == ["risque"]
    assert _deep(6).warnings == []
    assert _deep(7, feasibility=(1, True), legality=(3, False), risk=(1, False)).warnings == [
        "légalité", "faisabilité", "risque"]


def test_an_implausible_idea_is_no_longer_discarded_by_the_gate():
    implausible = GateVerdict(violates_red_line=False, illegal_or_harmful=False, plausible=False, score=8, reason="x")
    assert implausible.passed
    assert not GateVerdict(violates_red_line=True, illegal_or_harmful=False, plausible=True, score=8, reason="x").passed
    assert not GateVerdict(violates_red_line=False, illegal_or_harmful=True, plausible=True, score=8, reason="x").passed


def test_baseline_red_lines_always_in_prompt_even_when_user_gave_others():
    prompt = SPEC.to_prompt()
    assert "pas de médias" in prompt
    assert all(line in prompt for line in BASELINE_RED_LINES)


def test_bold_high_risk_solutions_are_added_to_report_selection():
    result = RunResult(run_dir=None, spec=SPEC, analysis=ANALYSIS)
    result.audited = [_deep(1), _deep(2), _deep(3, risk=(2, False), gate=9)]
    picked = [(rank, r.idea.id) for rank, r in ranked_for_report(result, top=1)]
    assert len(picked) == 2 and "I3" in {i for _, i in picked}


# --- interview -----------------------------------------------------------------------------------------------


async def test_interview_asks_then_builds_spec():
    asked = []

    async def ask(q):
        asked.append(q)
        return "5000 euros"

    spec = await run_interview(FakeLLM(), Settings(), "problème", ask, max_rounds=3)
    assert asked == ["Quel budget ?"]
    assert spec.title == "T"


# --- pipeline ------------------------------------------------------------------------------------------------


async def test_pipeline_funnel_and_outputs(tmp_path):
    settings = replace(Settings(), runs_dir=tmp_path, web_enabled=False)
    llm = FakeLLM()
    result = await solve(llm, settings, Console(quiet=True), SPEC, SolveParams(ideas_target=100, per_call=5, pool=10, top=3))

    run_dir = result.run_dir
    assert (run_dir / "report.md").read_text(encoding="utf-8").startswith("# T")
    assert (run_dir / "analysis.json").exists()
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["générées"] == 100
    assert stats["uniques"] <= 20  # 20 vecteurs distincts plafonnent la dédup
    assert stats["auditées"] <= 10
    # L'idée marquée nuisible par le filtre ne doit jamais atteindre l'audit, mais doit apparaître dans les écartées.
    audited_titles = [json.loads(line)["idea"]["title"] for line in (run_dir / "audits.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "idea 3" not in audited_titles
    # L'analyse de la situation est transmise à l'essaim.
    assert any("SITUATION ANALYSIS" in p for tag, p in llm.prompts if tag.startswith("essaim"))


# --- session interactive -------------------------------------------------------------------------------------


async def test_session_follow_up_commands(tmp_path):
    settings = replace(Settings(), runs_dir=tmp_path, web_enabled=False)
    llm = FakeLLM(buckets=1000)
    params = SolveParams(ideas_target=60, per_call=5, pool=5, top=3)
    result = await solve(llm, settings, Console(quiet=True), SPEC, params)
    audited_before, reserve_before = len(result.audited), len(result.reserve)
    assert reserve_before > 0

    script = iter(["1", "variantes 1 plus discret", "encore", "et si ça échoue ?", "rejetées", "n'importe quoi 99", "q"])

    async def read(_prompt):
        return next(script)

    console = Console(file=io.StringIO(), record=True, width=200)
    session = Session(llm, settings, console, result, params, read)
    await session.run()
    out = console.export_text()

    assert list(result.run_dir.glob("detail-*.md")), "le détail d'une solution doit être sauvegardé"
    assert any("plus discret" in p for tag, p in llm.prompts if tag.startswith("variantes")), "la consigne doit atteindre l'essaim"
    assert len(result.audited) > audited_before + 5  # variantes auditées + lot de réserve
    assert "réponse du conseiller" in out
    assert len(session.history) == 2  # la question libre et la commande inconnue partent au conseiller
    assert "Solutions retenues" in out


async def test_session_info_adds_fact_and_reruns(tmp_path):
    settings = replace(Settings(), runs_dir=tmp_path, web_enabled=False)
    llm = FakeLLM()
    params = SolveParams(ideas_target=20, per_call=5, pool=5, top=2)
    result = await solve(llm, settings, Console(quiet=True), SPEC, params)

    script = iter(["info elle part en congé maladie lundi", "q"])

    async def read(_prompt):
        return next(script)

    session = Session(llm, settings, Console(quiet=True), result, params, read)
    await session.run()
    assert "elle part en congé maladie lundi" in session.result.spec.key_facts


# --- verbose -------------------------------------------------------------------------------------------------


def test_verbose_writes_full_trace_to_file_but_not_to_screen_at_v1(tmp_path):
    log_file = tmp_path / "debug.log"
    screen = io.StringIO()
    setup_logging(Console(file=screen, width=200), verbosity=1, log_file=log_file)
    logger = logging.getLogger("swarm_solver")
    try:
        logger.getChild("llm").debug("appel court")
        logging.getLogger(TRACE).debug("PROMPT COMPLET")
    finally:
        for h in logger.handlers:
            h.close()
        logger.handlers.clear()

    assert "appel court" in screen.getvalue()
    assert "PROMPT COMPLET" not in screen.getvalue()
    assert "PROMPT COMPLET" in log_file.read_text(encoding="utf-8")


# --- l'outil doit rester générique --------------------------------------------------------------------------


def test_lenses_include_power_angles_only_when_someone_opposes_the_goal():
    from swarm_solver.ideation import ADVERSARIAL_LENSES, GENERAL_LENSES, select_lenses

    assert select_lenses(ANALYSIS.model_copy(update={"adversarial": False})) == GENERAL_LENSES
    mixed = select_lenses(ANALYSIS.model_copy(update={"adversarial": True}))
    assert set(mixed) == set(GENERAL_LENSES) | set(ADVERSARIAL_LENSES)
    assert mixed[1] in ADVERSARIAL_LENSES  # même un petit run voit les deux familles dès les premiers lots


def test_prompts_carry_no_vocabulary_from_a_single_use_case():
    """Régression: les prompts ont porté des traces du premier cas de test (conflit au travail). Ils servent à tout."""
    import importlib
    import re

    forbidden = re.compile(r"\b(colleagues?|interns?|dismiss\w*|HR|protectors?|hierarch\w*|fired|CSE|stagiaire)\b", re.I)
    modules = ["verify", "ideation", "interview", "analysis", "report", "session", "research.agent"]
    offenders = []
    for name in modules:
        module = importlib.import_module(f"swarm_solver.{name}")
        for attr, value in vars(module).items():
            if not attr.isupper():
                continue
            texts = value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value]
            for text in texts:
                if isinstance(text, str) and (m := forbidden.search(text)):
                    offenders.append(f"{name}.{attr}: {m.group(0)!r}")
    assert not offenders, offenders
