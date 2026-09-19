import asyncio
import io
import signal
import sys
import threading
from dataclasses import replace

import pytest
from rich.console import Console

from swarm_solver import cli
from swarm_solver.config import Settings
from swarm_solver.interrupts import Interrupted, UserQuit, interruptible, read_line
from swarm_solver.pipeline import RunResult, SolveParams, screen_and_audit, solve
from swarm_solver.session import Session
from swarm_solver.ui import Stepper, funnel_panel, score_bar, spec_panel

from test_pipeline import ANALYSIS, SPEC, FakeLLM


async def press_ctrl_c_then_hang(*_args, **_kwargs):
    """Simule un utilisateur qui appuie sur Ctrl+C pendant une étape longue."""
    signal.raise_signal(signal.SIGINT)
    await asyncio.sleep(30)


def render(renderable, width: int = 120) -> str:
    console = Console(file=io.StringIO(), width=width, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


# --- interruptible -------------------------------------------------------------------------------------------


async def test_interruptible_returns_the_result_when_nothing_happens():
    async def work():
        return 42

    assert await interruptible(work()) == 42


async def test_ctrl_c_cancels_only_the_wrapped_step_and_restores_the_handler():
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(Interrupted):
        await interruptible(press_ctrl_c_then_hang())
    assert signal.getsignal(signal.SIGINT) is previous


async def test_cancelling_from_outside_is_propagated_not_mistaken_for_ctrl_c():
    started = asyncio.Event()

    async def inner():
        started.set()
        await asyncio.sleep(30)

    outer = asyncio.create_task(interruptible(inner()))
    await started.wait()
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer


async def test_exceptions_of_the_step_pass_through():
    async def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        await interruptible(boom())


# --- lecture clavier -----------------------------------------------------------------------------------------


async def test_read_line_returns_the_typed_line_from_a_daemon_thread(monkeypatch):
    seen = {}

    def fake_input(prompt):
        seen["daemon"] = threading.current_thread().daemon
        return "oui"

    monkeypatch.setattr("builtins.input", fake_input)
    assert await read_line("> ") == "oui"
    assert seen["daemon"] is True  # un thread non-daemon empêcherait de quitter après Ctrl+C


async def test_read_line_propagates_end_of_input(monkeypatch):
    def fake_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    with pytest.raises(EOFError):
        await read_line("> ")


# --- pipeline: recherche interrompue --------------------------------------------------------------------------


async def test_interrupting_the_web_research_continues_without_it(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm_solver.pipeline.run_research", press_ctrl_c_then_hang)
    out = io.StringIO()
    settings = replace(Settings(), runs_dir=tmp_path)
    result = await solve(FakeLLM(), settings, Console(file=out, width=200), SPEC,
                         SolveParams(ideas_target=10, per_call=5, pool=3, top=2))

    assert result.research is None and (result.run_dir / "report.md").exists()
    assert "Recherche web interrompue" in out.getvalue()


async def test_declining_to_continue_after_interruption_quits(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm_solver.pipeline.run_research", press_ctrl_c_then_hang)
    asked = []

    async def refuse(question):
        asked.append(question)
        return False

    settings = replace(Settings(), runs_dir=tmp_path)
    with pytest.raises(UserQuit):
        await solve(FakeLLM(), settings, Console(quiet=True), SPEC, SolveParams(ideas_target=10, per_call=5),
                    ask_continue=refuse)
    assert asked and "sans le web" in asked[0]


async def test_accepting_to_continue_after_interruption_runs_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm_solver.pipeline.run_research", press_ctrl_c_then_hang)

    async def accept(question):
        return True

    settings = replace(Settings(), runs_dir=tmp_path)
    result = await solve(FakeLLM(), settings, Console(quiet=True), SPEC,
                         SolveParams(ideas_target=10, per_call=5, pool=3, top=2), ask_continue=accept)
    assert (result.run_dir / "report.md").exists()


async def test_interrupted_audit_leaves_the_result_untouched(tmp_path, monkeypatch):
    """Ctrl+C en plein audit ne doit ni perdre d'idées ni en compter à moitié."""

    async def cancelled(*_a, **_k):
        raise asyncio.CancelledError

    monkeypatch.setattr("swarm_solver.pipeline.deep_verify", cancelled)
    result = RunResult(run_dir=tmp_path, spec=SPEC, analysis=ANALYSIS)
    llm = FakeLLM(buckets=1000)
    from swarm_solver.ideation import generate_ideas

    ideas = await generate_ideas(llm, Settings(), Console(quiet=True), SPEC, ANALYSIS, target=10, per_call=5)
    with pytest.raises(asyncio.CancelledError):
        await screen_and_audit(llm, Settings(), Console(quiet=True), result, ideas, pool=3)
    assert result.dropped == [] and result.reserve == [] and result.audited == []


# --- session: une commande interrompue ramène au prompt -------------------------------------------------------


async def _finished_run(tmp_path):
    settings = replace(Settings(), runs_dir=tmp_path)
    llm, params = FakeLLM(buckets=1000), SolveParams(ideas_target=40, per_call=5, pool=4, top=2)
    return settings, llm, params, await solve(llm, replace(settings, web_enabled=False), Console(quiet=True), SPEC, params)


async def test_ctrl_c_during_a_command_returns_to_the_prompt(tmp_path, monkeypatch):
    settings, llm, params, result = await _finished_run(tmp_path)
    monkeypatch.setattr("swarm_solver.session.ask_web", press_ctrl_c_then_hang)
    script = iter(["recherche une question", "liste", "q"])

    async def read(_):
        return next(script)

    console = Console(file=io.StringIO(), record=True, width=200)
    await Session(llm, replace(settings, web_enabled=True), console, result, params, read).run()

    out = console.export_text()
    assert "Commande interrompue" in out
    assert list(script) == []  # la session a continué et lu "liste" puis "q"


async def test_interrupting_encore_keeps_the_batch_in_reserve(tmp_path, monkeypatch):
    settings, llm, params, result = await _finished_run(tmp_path)
    reserve_before, audited_before = len(result.reserve), len(result.audited)
    assert reserve_before > 0
    monkeypatch.setattr("swarm_solver.session.deep_verify", press_ctrl_c_then_hang)
    script = iter(["encore", "q"])

    async def read(_):
        return next(script)

    await Session(llm, settings, Console(quiet=True), result, params, read).run()
    assert len(result.reserve) == reserve_before and len(result.audited) == audited_before


# --- CLI ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [KeyboardInterrupt, UserQuit])
def test_cli_exits_with_130_and_no_traceback_on_interruption(monkeypatch, capsys, exc):
    async def interrupted(args, settings):
        raise exc

    monkeypatch.setattr(cli, "cmd_solve", interrupted)
    monkeypatch.setattr(sys, "argv", ["swarm-solver", "solve", "--no-web"])
    with pytest.raises(SystemExit) as info:
        cli.main()
    assert info.value.code == 130


# --- interface -----------------------------------------------------------------------------------------------


def test_score_bar_length_and_colour_thresholds():
    assert score_bar(10).plain.startswith("█" * 10)
    assert score_bar(0).plain.startswith("░" * 10)
    assert [str(score_bar(s).style) for s in (8, 5.5, 2)] == ["green", "yellow", "red"]  # couleur de la partie remplie


def test_stepper_numbers_stages_and_records_durations():
    console = Console(file=io.StringIO(), width=100, record=True)
    stepper = Stepper(console, total=3)
    stepper.next("Analyse")
    stepper.next("Essaim")
    timings = stepper.finish()
    out = console.export_text()
    assert "Étape 1/3" in out and "Étape 2/3" in out and "Essaim" in out
    assert [t for t, _ in timings] == ["Analyse", "Essaim"] and all(s >= 0 for _, s in timings)


def test_funnel_panel_shows_every_stage_with_its_value_and_the_total_time():
    stats = {"générées": 100, "uniques": 60, "auditées": 10, "retenues": 4}
    out = render(funnel_panel(stats, [("Essaim", 125.0), ("Rapport", 30.0)]))
    for label in stats:
        assert label in out
    assert "100" in out and "2 min 05 s" in out and "2 min 35 s" in out  # total: 155 s


def test_spec_panel_shows_the_users_limits_but_not_the_hardcoded_baseline():
    spec = SPEC.model_copy(update={"red_lines": ["pas de médias"], "key_facts": ["fait [important]"]})
    out = render(spec_panel(spec))
    assert "pas de médias" in out and "fait [important]" in out  # crochets: pas de balisage rich accidentel
    assert "fabricated or stolen evidence" not in out
