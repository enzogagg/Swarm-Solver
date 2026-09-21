import io
import json
from dataclasses import replace

from rich.console import Console

from swarm_solver.config import Settings
from swarm_solver.ideation import RULES
from swarm_solver.pipeline import RunResult, SolveParams, solve
from swarm_solver.report import _render, write_report
from swarm_solver.schemas import GateVerdict
from swarm_solver.session import Session
from swarm_solver.verify import gate

from test_pipeline import ANALYSIS, SPEC, FakeLLM, _deep, _idea
from test_resume import NoCallsLLM

PARAMS = SolveParams(ideas_target=40, per_call=5, pool=4, top=2)


def make_result(tmp_path) -> RunResult:
    result = RunResult(run_dir=tmp_path, spec=SPEC, analysis=ANALYSIS)
    result.audited = [
        _deep(1, gate=9),                                    # classée
        _deep(2, gate=8),                                    # classée
        _deep(3, legality=(2, True), gate=7),                # écartée par l'audit (illégal avéré)
    ]
    result.dropped = [(_idea(9), "repose sur un acte illégal")]  # écartée par le filtre
    return result


# --- décisions de l'utilisateur -----------------------------------------------------------------------------


def test_the_user_can_remove_a_ranked_idea_and_bring_it_back(tmp_path):
    result = make_result(tmp_path)
    assert [r.idea.id for r in result.survivors] == ["I1", "I2"]

    result.veto("I1")
    assert [r.idea.id for r in result.survivors] == ["I2"]
    item = next(it for it in result.rejected_items() if it.idea.id == "I1")
    assert item.origin == "toi"

    result.keep(item)
    assert [r.idea.id for r in result.survivors] == ["I1", "I2"]


def test_the_user_overrides_the_audits_verdict_on_an_idea_flagged_illegal(tmp_path):
    result = make_result(tmp_path)
    item = next(it for it in result.rejected_items() if it.origin == "audit")
    assert item.idea.id == "I3" and item.reason.startswith("illégal")

    result.keep(item)
    ranked = [r for r in result.survivors if r.idea.id == "I3"]
    assert ranked, "l'utilisateur a décidé: l'idée rejoint le classement"
    assert "légalité" in ranked[0].warnings, "mais l'alerte reste visible à côté"


def test_an_idea_dropped_by_the_gate_needs_its_audit_to_join_the_ranking(tmp_path):
    import pytest

    result = make_result(tmp_path)
    item = next(it for it in result.rejected_items() if it.origin == "filtre")
    with pytest.raises(ValueError):
        result.keep(item)

    result.keep(item, _deep(9))
    assert "I9" in [r.idea.id for r in result.survivors] and result.dropped == []


def test_rejected_items_are_listed_with_stable_references(tmp_path):
    result = make_result(tmp_path)
    result.veto("I2")
    assert [(it.idea.id, it.origin) for it in result.rejected_items()] == [
        ("I3", "audit"), ("I2", "toi"), ("I9", "filtre")]


# --- session ---------------------------------------------------------------------------------------------------


async def finished_run(tmp_path):
    settings = replace(Settings(), runs_dir=tmp_path, web_enabled=False)
    llm = FakeLLM(buckets=1000)
    return settings, llm, await solve(llm, settings, Console(quiet=True), SPEC, PARAMS)


async def run_commands(settings, llm, result, commands):
    script = iter([*commands, "q"])

    async def read(_):
        return next(script)

    console = Console(file=io.StringIO(), record=True, width=220)
    await Session(llm, settings, console, result, PARAMS, read).run()
    return console.export_text()


async def test_the_session_lists_numbered_rejections_and_lets_the_user_restore_one(tmp_path):
    settings, llm, result = await finished_run(tmp_path)
    assert result.dropped, "le faux filtre écarte l'idée jugée illégale"
    dropped_id = result.dropped[0][0].id

    out = await run_commands(settings, llm, result, ["rejetées", f"garder {dropped_id}"])

    assert "Hors classement" in out and "R1" in out and "Tu décides" in out
    assert "Remise dans le classement" in out
    assert dropped_id in [r.idea.id for r in result.survivors]  # auditée à la demande, puis classée
    assert dropped_id not in [i.id for i, _ in result.dropped]


async def test_the_session_can_remove_a_ranked_idea_and_restore_it_with_its_reference(tmp_path):
    settings, llm, result = await finished_run(tmp_path)
    top = result.survivors[0].idea.id

    await run_commands(settings, llm, result, ["écarter 1"])
    assert top not in [r.idea.id for r in result.survivors]

    refs = [it.idea.id for it in result.rejected_items()]
    await run_commands(settings, llm, result, [f"garder R{refs.index(top) + 1}"])
    assert top in [r.idea.id for r in result.survivors]


async def test_an_unknown_reference_is_reported_without_changing_anything(tmp_path):
    settings, llm, result = await finished_run(tmp_path)
    before = len(result.survivors)
    out = await run_commands(settings, llm, result, ["garder R999", "écarter 999"])
    assert "Référence inconnue" in out and "Numéro invalide" in out
    assert len(result.survivors) == before


async def test_decisions_survive_a_resume(tmp_path):
    settings, llm, result = await finished_run(tmp_path)
    top = result.survivors[0].idea.id
    await run_commands(settings, llm, result, ["écarter 1"])
    assert json.loads((result.run_dir / "decisions.json").read_text(encoding="utf-8"))["vetoed"] == [top]

    again = await solve(NoCallsLLM(), settings, Console(quiet=True), SPEC, PARAMS, resume_dir=result.run_dir)
    assert top in again.vetoed and top not in [r.idea.id for r in again.survivors]


# --- filtre et prompts --------------------------------------------------------------------------------------


class ImplausibleGateLLM(FakeLLM):
    async def structured(self, model, system, user, schema, **kwargs):
        return GateVerdict(violates_red_line=False, illegal_or_harmful=False, plausible=False, score=9, reason="ambitieux")


async def test_an_implausible_idea_stays_in_play_with_a_capped_score():
    outcome = await gate(ImplausibleGateLLM(), Settings(diversity=0), Console(quiet=True), SPEC, [_idea(1), _idea(2)], keep=5)
    assert outcome.dropped == []
    assert [v.score for _, v in outcome.kept] == [4, 4]  # gardée, mais plus en tête du classement


def test_the_swarm_is_no_longer_restricted_to_what_the_user_can_do_alone():
    assert "Do not propose anything that needs an authority" not in RULES
    assert "Do not limit yourself to what the user could do alone" in RULES
    assert "committees" not in RULES


async def test_the_report_receives_the_flags_and_tells_the_user_to_decide():
    flagged = _deep(1, legality=(3, False), risk=(2, False))
    assert "flags=légalité, risque" in _render([(1, flagged)])

    llm = FakeLLM()
    await write_report(llm, Settings(), SPEC, ANALYSIS, [(1, flagged)], [], {"générées": 1})
    prompt = next(p for tag, p in llm.prompts if tag == "rapport")
    assert "never hide or drop a flagged solution" in prompt and "`garder`" in prompt
