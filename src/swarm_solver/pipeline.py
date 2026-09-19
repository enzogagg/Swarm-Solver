import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console

from .analysis import analyze
from .config import Settings
from .dedup import dedupe_ideas
from .ideation import generate_ideas
from .llm import LLMClient
from .logs import LOG
from .report import write_report
from .research import Research, ResearchBrief, run_research
from .schemas import Analysis, DeepResult, GateVerdict, Idea, ProblemSpec
from .verify import deep_verify, gate

log = logging.getLogger(f"{LOG}.pipeline")


@dataclass
class SolveParams:
    ideas_target: int = 1000
    per_call: int = 10
    pool: int = 60  # idées envoyées à l'audit approfondi à chaque lot
    top: int = 8  # solutions détaillées dans le rapport


@dataclass
class RunResult:
    """État d'un run. La session interactive l'enrichit (variantes, lots suivants)."""

    run_dir: Path
    spec: ProblemSpec
    analysis: Analysis
    research: Research | None = None
    raw_count: int = 0
    unique_count: int = 0
    audited: list[DeepResult] = field(default_factory=list)
    dropped: list[tuple[Idea, str]] = field(default_factory=list)  # écartées par le filtre: (idée, raison)
    reserve: list[tuple[Idea, GateVerdict]] = field(default_factory=list)  # ont passé le filtre, pas encore auditées

    @property
    def brief(self) -> ResearchBrief | None:
        return self.research.brief if self.research else None

    @property
    def survivors(self) -> list[DeepResult]:
        return sorted((r for r in self.audited if r.rejection is None), key=lambda r: r.total, reverse=True)

    @property
    def audit_rejects(self) -> list[DeepResult]:
        return [r for r in self.audited if r.rejection is not None]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "générées": self.raw_count,
            "uniques": self.unique_count,
            "écartées par le filtre": len(self.dropped),
            "auditées": len(self.audited),
            "retenues": len(self.survivors),
            "en réserve": len(self.reserve),
        }

    def take_reserve(self, n: int) -> list[tuple[Idea, GateVerdict]]:
        self.reserve.sort(key=lambda iv: iv[1].score, reverse=True)
        batch, self.reserve = self.reserve[:n], self.reserve[n:]
        return batch

    def save(self) -> None:
        _dump(self.run_dir / "audits.jsonl", self.audited)
        (self.run_dir / "dropped.jsonl").write_text(
            "\n".join(json.dumps({"id": i.id, "title": i.title, "reason": why}, ensure_ascii=False) for i, why in self.dropped),
            encoding="utf-8",
        )
        (self.run_dir / "stats.json").write_text(json.dumps(self.stats, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.research:
            (self.run_dir / "research.json").write_text(self.research.model_dump_json(indent=2), encoding="utf-8")


def _dump(path: Path, items: list[BaseModel]) -> None:
    path.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")


async def screen_and_audit(
    llm: LLMClient, settings: Settings, console: Console, result: RunResult, ideas: list[Idea], pool: int
) -> None:
    """Filtre rapide sur `ideas`, puis audit approfondi des meilleures. Met à jour `result`."""
    outcome = await gate(llm, settings, console, result.spec, ideas, keep=pool)
    result.dropped += [(i, v.reason) for i, v in outcome.dropped]
    result.reserve += outcome.reserve
    result.audited += await deep_verify(llm, settings, console, result.spec, outcome.kept, result.brief)


def ranked_for_report(result: RunResult, top: int) -> list[tuple[int, DeepResult]]:
    """Les `top` meilleures, plus les 3 "coups audacieux" (risque élevé, fort gain attendu) s'ils n'y sont pas."""
    survivors = list(enumerate(result.survivors, 1))
    chosen = survivors[:top]
    bold = sorted((s for s in survivors[top:] if s[1].risk_level == "high"), key=lambda s: s[1].gate_score, reverse=True)
    return chosen + bold[:3]


async def solve(
    llm: LLMClient, settings: Settings, console: Console, spec: ProblemSpec, params: SolveParams | None = None
) -> RunResult:
    """Entonnoir : recherche web -> analyse -> essaim -> dédup -> filtre rapide -> audit approfondi -> rapport."""
    params = params or SolveParams()
    run_dir = settings.runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "problem.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    research = None
    if settings.web_enabled:
        try:
            research = await run_research(llm, settings, console, spec)
        except Exception as e:  # noqa: BLE001 - réseau coupé ou fournisseur mal configuré: on continue sans le web
            console.print(f"[yellow]Recherche web indisponible ({type(e).__name__}: {e}). On continue sans.[/yellow]")
            log.debug("recherche web échouée", exc_info=True)

    with console.status("Analyse de la situation: acteurs, rapports de force, leviers (modèle solide)..."):
        analysis = await analyze(llm, settings, spec, research.brief if research else None)
    (run_dir / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    log.debug("analyse: %d acteurs, %d leviers", len(analysis.actors), len(analysis.leverage_points))

    result = RunResult(run_dir=run_dir, spec=spec, analysis=analysis, research=research)
    raw = await generate_ideas(
        llm, settings, console, spec, analysis, params.ideas_target, params.per_call, result.brief
    )
    result.raw_count = len(raw)
    _dump(run_dir / "ideas_raw.jsonl", raw)

    console.print(f"Déduplication de {len(raw)} idées...")
    unique = await dedupe_ideas(llm, settings, raw)
    result.unique_count = len(unique)
    _dump(run_dir / "ideas_unique.jsonl", unique)

    await screen_and_audit(llm, settings, console, result, unique, params.pool)
    result.save()

    survivors = result.survivors
    if not survivors:
        console.print("[red]Aucune solution n'a survécu aux audits. Lance `encore`, ou ajoute des informations.[/red]")
        return result

    rejected = [f"{i.title}: {why}" for i, why in result.dropped] + [
        f"{r.idea.title}: {r.rejection}" for r in result.audit_rejects
    ]
    with console.status("Rédaction du rapport (modèle solide)..."):
        report = await write_report(
            llm, settings, spec, analysis, ranked_for_report(result, params.top), rejected, result.stats, research
        )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return result
