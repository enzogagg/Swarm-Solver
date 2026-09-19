"""Éléments visuels du terminal (rich) : bandeau, étapes, fiche, barres de score, entonnoir."""

import time

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Settings
from .schemas import ProblemSpec

BAR_WIDTH = 10


def score_color(score: float) -> str:
    return "green" if score >= 7 else "yellow" if score >= 5 else "red"


def score_bar(score: float, width: int = BAR_WIDTH) -> Text:
    """Barre de 0 à 10 colorée: verte à partir de 7, jaune à partir de 5, rouge en dessous."""
    filled = max(0, min(width, round(score / 10 * width)))
    bar = Text("█" * filled, style=score_color(score))
    bar.append("░" * (width - filled), style="grey37")
    bar.append(f" {score:.1f}", style="bold")
    return bar


def banner(console: Console, settings: Settings) -> None:
    web = "[green]activée[/green]" if settings.web_enabled else "[dim]désactivée[/dim]"
    body = Text.from_markup(
        f"[dim]essaim[/dim] {settings.fast_model}   [dim]audits[/dim] {settings.strong_model}   "
        f"[dim]web[/dim] {web}   [dim]droit[/dim] {settings.jurisdiction}\n"
        "[dim]Ctrl+C interrompt l'étape en cours (recherche web: tu choisis de continuer sans).[/dim]"
    )
    console.print(
        Panel(body, title="[bold cyan]Swarm Solver[/bold cyan]", subtitle="agents locaux via Ollama",
              border_style="cyan", box=box.ROUNDED, padding=(0, 2))
    )


class Stepper:
    """Numérote les étapes du pipeline ("Étape 3/7") et mesure leur durée."""

    def __init__(self, console: Console, total: int):
        self.console, self.total = console, total
        self.done: list[tuple[str, float]] = []
        self._title: str | None = None
        self._started = 0.0

    def next(self, title: str) -> None:
        self._close()
        self._title, self._started = title, time.perf_counter()
        self.console.print()
        self.console.rule(f"[bold cyan]Étape {len(self.done) + 1}/{self.total}[/bold cyan] · {escape(title)}", style="cyan")

    def finish(self) -> list[tuple[str, float]]:
        self._close()
        return self.done

    def _close(self) -> None:
        if self._title is not None:
            self.done.append((self._title, time.perf_counter() - self._started))
            self._title = None


def duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes} min {sec:02d} s" if minutes else f"{sec} s"


def spec_panel(spec: ProblemSpec) -> Panel:
    """La fiche problème lisible d'un coup d'œil (les limites de base, codées en dur, ne sont pas répétées ici)."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True, justify="right")
    grid.add_column()

    def row(label: str, value: str | list[str]) -> None:
        if not value:
            return
        text = value if isinstance(value, str) else "\n".join(f"• {v}" for v in value)
        grid.add_row(label, escape(text))

    row("Objectif", spec.underlying_goal)
    row("Contexte", spec.context)
    row("Faits", spec.key_facts)
    row("Déjà tenté", spec.already_tried)
    row("Contraintes", spec.constraints)
    row("Ressources", spec.resources)
    row("Tes limites", spec.red_lines)
    row("Succès", spec.success_criteria)
    return Panel(grid, title=f"[bold]Fiche problème[/bold] · {escape(spec.title)}", border_style="blue", padding=(1, 2))


def report_panel(markdown: str) -> Panel:
    return Panel(Markdown(markdown), title="[bold]Rapport[/bold]", border_style="green", padding=(1, 2))


def funnel_panel(stats: dict[str, int], timings: list[tuple[str, float]]) -> Panel:
    """L'entonnoir en barres proportionnelles aux idées générées, et le temps passé par étape."""
    top = max(stats.get("générées", 0), 1)
    funnel = Table.grid(padding=(0, 1))
    funnel.add_column(justify="right", no_wrap=True)
    funnel.add_column(no_wrap=True)
    funnel.add_column(justify="right", style="bold")
    styles = {"générées": "blue", "uniques": "blue", "auditées": "cyan", "retenues": "green",
              "écartées par le filtre": "red", "en réserve": "yellow"}
    for label, value in stats.items():
        width = max(1, round(value / top * 24)) if value else 0
        funnel.add_row(label, Text("█" * width, style=styles.get(label, "white")), str(value))

    parts: list[RenderableType] = [funnel]
    if timings:
        times = Table.grid(padding=(0, 2))
        times.add_column(style="dim")
        times.add_column(justify="right")
        for title, seconds in timings:
            times.add_row(escape(title), duration(seconds))
        times.add_row("[bold]Total[/bold]", f"[bold]{duration(sum(s for _, s in timings))}[/bold]")
        parts += [Text(""), times]
    return Panel(Group(*parts), title="[bold]Entonnoir et durées[/bold]", border_style="magenta", padding=(1, 2))
