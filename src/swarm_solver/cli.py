import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from ollama import AsyncClient
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import Settings
from .interrupts import UserQuit, read_line
from .interview import run_interview
from .llm import OllamaLLM
from .logs import setup_logging
from .pipeline import SolveParams, solve
from .schemas import ProblemSpec
from .session import Session
from .ui import banner, funnel_panel, report_panel, spec_panel

console = Console()
_read = read_line


async def _ask(question: str) -> str:
    console.print(f"[bold cyan]?[/bold cyan] {escape(question)}")
    return (await _read("  > ")).strip() or "(pas de réponse)"


async def _confirm(question: str) -> bool:
    return (await _read(f"{question} [O/n] ")).strip().lower() not in {"n", "non"}


async def cmd_solve(args: argparse.Namespace, settings: Settings) -> int:
    banner(console, settings)
    llm = OllamaLLM(settings)
    if args.spec:
        spec = ProblemSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
    else:
        problem = args.problem or (await _read("Décris ton problème :\n> ")).strip()
        if not problem:
            console.print("[red]Problème vide.[/red]")
            return 1
        spec = await run_interview(llm, settings, problem, _ask, max_rounds=args.rounds)
        console.print(spec_panel(spec))
        if not await _confirm("Lancer la résolution ?"):
            return 0

    return await _run_and_discuss(args, settings, llm, spec)


async def _run_and_discuss(
    args: argparse.Namespace, settings: Settings, llm: OllamaLLM, spec: ProblemSpec, resume_dir: Path | None = None
) -> int:
    """Lance (ou reprend) le pipeline, affiche le rapport, puis ouvre la session interactive."""
    params = SolveParams(ideas_target=args.ideas, per_call=args.per_call, pool=args.pool, top=args.top)
    result = await solve(llm, settings, console, spec, params, _confirm, resume_dir=resume_dir)
    report = result.run_dir / "report.md"
    if report.exists():
        console.print(report_panel(report.read_text(encoding="utf-8")))
    console.print(funnel_panel(result.stats, result.timings))
    console.print(f"Résultats complets dans [bold]{result.run_dir}[/bold]\n")

    if not args.no_followup:
        await Session(llm, settings, console, result, params, _read).run()
    return 0


def find_run(name: str | None, runs_dir: Path) -> Path:
    """Dossier d'un run: chemin explicite, nom sous `runs/`, ou le plus récent (`latest` ou rien)."""
    if name in (None, "latest"):
        candidates = sorted((d for d in runs_dir.glob("2*") if (d / "problem.json").exists()), key=lambda d: d.name)
        if not candidates:
            raise FileNotFoundError(f"Aucun run dans {runs_dir}")
        return candidates[-1]
    for candidate in (Path(name), runs_dir / name):
        if (candidate / "problem.json").exists():
            return candidate
    raise FileNotFoundError(f"Run introuvable: {name} (il faut un dossier contenant problem.json)")


async def cmd_resume(args: argparse.Namespace, settings: Settings) -> int:
    banner(console, settings)
    try:
        run_dir = find_run(args.run, settings.runs_dir)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    spec = ProblemSpec.model_validate_json((run_dir / "problem.json").read_text(encoding="utf-8"))
    console.print(spec_panel(spec))
    return await _run_and_discuss(args, settings, OllamaLLM(settings), spec, resume_dir=run_dir)


async def cmd_doctor(_: argparse.Namespace, settings: Settings) -> int:
    try:
        installed = {m.model for m in (await AsyncClient(host=settings.host).list()).models}
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Ollama injoignable sur {settings.host}: {e}[/red]")
        return 1
    ok = True
    table = Table(title="Modèles Ollama", box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    for col in ("Rôle", "Modèle", "État"):
        table.add_column(col)
    for role, model in [("rapide", settings.fast_model), ("solide", settings.strong_model), ("embeddings", settings.embed_model)]:
        present = model in installed or f"{model}:latest" in installed
        ok &= present
        table.add_row(role, model, "[green]présent[/green]" if present else f"[red]manquant[/red]  ollama pull {model}")
    console.print(table)
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="swarm-solver", description="Résolveur de problèmes multi-agents local")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: appels, durées, verdicts. -vv: prompts et réponses complets. Journal dans runs/debug.log")
    sub = parser.add_subparsers(dest="command", required=True)

    tuning = argparse.ArgumentParser(add_help=False)
    tuning.add_argument("--ideas", type=int, default=1000, help="idées brutes à générer")
    tuning.add_argument("--per-call", type=int, default=10, help="idées demandées par appel")
    tuning.add_argument("--pool", type=int, default=60, help="idées envoyées à l'audit approfondi à chaque lot")
    tuning.add_argument("--top", type=int, default=8, help="solutions dans le rapport")
    tuning.add_argument("--no-followup", action="store_true", help="pas de session interactive après le rapport")
    tuning.add_argument("--no-web", action="store_true", help="aucune recherche sur internet (100 %% hors ligne)")
    tuning.add_argument("--fast-model")
    tuning.add_argument("--strong-model")
    tuning.add_argument("--concurrency", type=int)

    p = sub.add_parser("solve", parents=[common, tuning], help="interview puis résolution")
    p.add_argument("problem", nargs="?", help="description initiale du problème")
    p.add_argument("--spec", help="fiche problème JSON existante (saute l'interview)")
    p.add_argument("--rounds", type=int, default=3, help="tours de questions max")

    r = sub.add_parser("resume", parents=[common, tuning],
                       help="reprend un run là où il s'était arrêté et rouvre la session interactive")
    r.add_argument("run", nargs="?", help="dossier du run (ex: runs/20260919-165549) ou son nom ; défaut: le plus récent")
    sub.add_parser("doctor", parents=[common], help="vérifie Ollama et les modèles")

    args = parser.parse_args()
    settings = Settings.from_env()
    overrides = {
        k: v
        for k, v in {
            "fast_model": getattr(args, "fast_model", None),
            "strong_model": getattr(args, "strong_model", None),
            "concurrency": getattr(args, "concurrency", None),
        }.items()
        if v
    }
    if getattr(args, "no_web", False):
        overrides["web_enabled"] = False
    settings = replace(settings, **overrides)
    setup_logging(console, args.verbose, settings.runs_dir / "debug.log")
    handler = {"solve": cmd_solve, "resume": cmd_resume, "doctor": cmd_doctor}[args.command]
    try:
        code = asyncio.run(handler(args, settings))
    except (KeyboardInterrupt, UserQuit):
        console.print("\n[yellow]Interrompu. Ce qui a été produit est conservé dans le dossier du run.[/yellow]")
        code = 130
    sys.exit(code)
