import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from ollama import AsyncClient
from rich.console import Console
from rich.markdown import Markdown

from .config import Settings
from .interview import run_interview
from .llm import OllamaLLM
from .logs import setup_logging
from .pipeline import SolveParams, solve
from .schemas import ProblemSpec
from .session import Session

console = Console()


async def _read(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def _ask(question: str) -> str:
    console.print(f"[bold cyan]?[/bold cyan] {question}")
    return (await _read("  > ")).strip() or "(pas de réponse)"


async def cmd_solve(args: argparse.Namespace, settings: Settings) -> int:
    llm = OllamaLLM(settings)
    if args.spec:
        spec = ProblemSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
    else:
        problem = args.problem or (await _read("Décris ton problème :\n> ")).strip()
        if not problem:
            console.print("[red]Problème vide.[/red]")
            return 1
        spec = await run_interview(llm, settings, problem, _ask, max_rounds=args.rounds)
        console.print(Markdown(f"**Fiche problème**\n\n```\n{spec.to_prompt()}\n```"))
        if (await _read("Lancer la résolution ? [O/n] ")).strip().lower() in {"n", "non"}:
            return 0

    params = SolveParams(ideas_target=args.ideas, per_call=args.per_call, pool=args.pool, top=args.top)
    result = await solve(llm, settings, console, spec, params)
    report = result.run_dir / "report.md"
    if report.exists():
        console.print(Markdown(report.read_text(encoding="utf-8")))
    console.print(f"\nRésultats complets dans [bold]{result.run_dir}[/bold]")

    if not args.no_followup:
        await Session(llm, settings, console, result, params, _read).run()
    return 0


async def cmd_doctor(_: argparse.Namespace, settings: Settings) -> int:
    try:
        installed = {m.model for m in (await AsyncClient(host=settings.host).list()).models}
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Ollama injoignable sur {settings.host}: {e}[/red]")
        return 1
    ok = True
    for role, model in [("rapide", settings.fast_model), ("solide", settings.strong_model), ("embeddings", settings.embed_model)]:
        present = model in installed or f"{model}:latest" in installed
        ok &= present
        console.print(f"{'[green]OK[/green]' if present else '[red]MANQUANT[/red]'}  {role}: {model}"
                      + ("" if present else f"   ->  ollama pull {model}"))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="swarm-solver", description="Résolveur de problèmes multi-agents local")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: appels, durées, verdicts. -vv: prompts et réponses complets. Journal dans runs/debug.log")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("solve", parents=[common], help="interview puis résolution")
    p.add_argument("problem", nargs="?", help="description initiale du problème")
    p.add_argument("--spec", help="fiche problème JSON existante (saute l'interview)")
    p.add_argument("--rounds", type=int, default=3, help="tours de questions max")
    p.add_argument("--ideas", type=int, default=1000, help="idées brutes à générer")
    p.add_argument("--per-call", type=int, default=10, help="idées demandées par appel")
    p.add_argument("--pool", type=int, default=60, help="idées envoyées à l'audit approfondi à chaque lot")
    p.add_argument("--top", type=int, default=8, help="solutions dans le rapport")
    p.add_argument("--no-followup", action="store_true", help="pas de session interactive après le rapport")
    p.add_argument("--no-web", action="store_true", help="aucune recherche sur internet (100 %% hors ligne)")
    p.add_argument("--fast-model")
    p.add_argument("--strong-model")
    p.add_argument("--concurrency", type=int)
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
    handler = {"solve": cmd_solve, "doctor": cmd_doctor}[args.command]
    sys.exit(asyncio.run(handler(args, settings)))
