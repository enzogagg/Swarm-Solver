import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from rich.console import Console

from .analysis import analyze
from .config import Settings
from .dedup import dedupe_ideas
from .ideation import generate_ideas
from .interrupts import Interrupted, UserQuit, interruptible
from .llm import LLMClient
from .logs import LOG
from .report import write_report
from .research import Research, ResearchBrief, complete_brief, run_research
from .schemas import Analysis, DeepResult, GateVerdict, Idea, ProblemSpec
from .ui import Stepper
from .verify import deep_verify, gate

log = logging.getLogger(f"{LOG}.pipeline")

M = TypeVar("M", bound=BaseModel)


class ReserveEntry(BaseModel):
    """Une idée qui a passé le filtre mais n'a pas encore été auditée, avec la note que le filtre lui a donnée."""

    idea: Idea
    verdict: GateVerdict


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
    timings: list[tuple[str, float]] = field(default_factory=list)  # (étape, secondes)

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

    def peek_reserve(self, n: int) -> list[tuple[Idea, GateVerdict]]:
        """Les `n` meilleures idées en réserve, SANS les retirer: à retirer une fois leur audit terminé, pour qu'une
        interruption (Ctrl+C) en plein audit ne les fasse pas disparaître."""
        self.reserve.sort(key=lambda iv: iv[1].score, reverse=True)
        return self.reserve[:n]

    def save(self) -> None:
        _dump(self.run_dir / "audits.jsonl", self.audited)
        (self.run_dir / "dropped.jsonl").write_text(
            "\n".join(json.dumps({"id": i.id, "title": i.title, "reason": why}, ensure_ascii=False) for i, why in self.dropped),
            encoding="utf-8",
        )
        _dump(self.run_dir / "reserve.jsonl", [ReserveEntry(idea=i, verdict=v) for i, v in self.reserve])
        (self.run_dir / "stats.json").write_text(json.dumps(self.stats, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.research:
            (self.run_dir / "research.json").write_text(self.research.model_dump_json(indent=2), encoding="utf-8")


def _dump(path: Path, items: list[BaseModel]) -> None:
    path.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")


async def screen_and_audit(
    llm: LLMClient,
    settings: Settings,
    console: Console,
    result: RunResult,
    ideas: list[Idea],
    pool: int,
    stepper: Stepper | None = None,
) -> None:
    """Filtre rapide sur `ideas`, puis audit approfondi des meilleures. Met à jour `result`."""
    if stepper:
        stepper.next("Filtre rapide de chaque idée")
    outcome = await gate(llm, settings, console, result.spec, ideas, keep=pool)
    if stepper:
        stepper.next("Audit approfondi (faisabilité, risques, légalité)")
    audited = await deep_verify(llm, settings, console, result.spec, outcome.kept, result.brief)
    # Toutes les mises à jour à la fin: une interruption (Ctrl+C) en cours de route laisse `result` intact.
    result.dropped += [(i, v.reason) for i, v in outcome.dropped]
    result.reserve += outcome.reserve
    result.audited += audited


def ranked_for_report(result: RunResult, top: int) -> list[tuple[int, DeepResult]]:
    """Les `top` meilleures, plus les 3 "coups audacieux" (risque élevé, fort gain attendu) s'ils n'y sont pas."""
    survivors = list(enumerate(result.survivors, 1))
    chosen = survivors[:top]
    bold = sorted((s for s in survivors[top:] if s[1].risk_level == "high"), key=lambda s: s[1].gate_score, reverse=True)
    return chosen + bold[:3]


AskContinue = Callable[[str], Awaitable[bool]]


def _read_model(path: Path, model: type[M]) -> M | None:
    """Un artefact JSON du run, ou None s'il manque ou n'a plus le format actuel (l'étape sera refaite)."""
    if not path.exists():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        log.debug("%s n'a plus le format actuel: l'étape sera refaite", path.name)
        return None


def _read_lines(path: Path, model: type[M], console: Console | None = None) -> list[M] | None:
    """Les lignes JSON d'un artefact. None s'il manque ou est vide (étape à refaire).

    Une ligne qui n'a plus le format actuel (run ancien) est ignorée, et signalée.
    """
    if not path.exists():
        return None
    items, skipped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(model.model_validate_json(line))
        except ValidationError:
            skipped += 1
    if skipped and console:
        console.print(f"[yellow]{skipped} ligne(s) de {path.name} ignorée(s): format d'une ancienne version.[/yellow]")
    return items or None


def _restore_audit(result: RunResult, unique: list[Idea], audited: list[DeepResult]) -> None:
    """Reconstruit l'état d'après-audit d'un run: idées écartées par le filtre, et idées encore en réserve."""
    by_id = {i.id: i for i in unique}
    result.audited = audited
    dropped_path = result.run_dir / "dropped.jsonl"
    rows = [json.loads(line) for line in dropped_path.read_text(encoding="utf-8").splitlines() if line.strip()] if dropped_path.exists() else []
    result.dropped = [(by_id[r["id"]], r["reason"]) for r in rows if r["id"] in by_id]
    entries = _read_lines(result.run_dir / "reserve.jsonl", ReserveEntry)
    if entries is None:  # run ancien: la note du filtre n'avait pas été sauvegardée
        seen = {r.idea.id for r in audited} | {i.id for i, _ in result.dropped}
        neutral = GateVerdict(violates_red_line=False, illegal_or_harmful=False, plausible=True, score=5,
                              reason="note du filtre non conservée")
        entries = [ReserveEntry(idea=i, verdict=neutral) for i in unique if i.id not in seen]
    result.reserve = [(e.idea, e.verdict) for e in entries]


async def solve(
    llm: LLMClient,
    settings: Settings,
    console: Console,
    spec: ProblemSpec,
    params: SolveParams | None = None,
    ask_continue: AskContinue | None = None,
    resume_dir: Path | None = None,
) -> RunResult:
    """Entonnoir : recherche web -> analyse -> essaim -> dédup -> filtre rapide -> audit approfondi -> rapport.

    Ctrl+C pendant la recherche web l'interrompt seule: `ask_continue` demande alors s'il faut poursuivre sans le web
    (sans callback, on poursuit). Répondre non lève `UserQuit`. Ailleurs, Ctrl+C arrête le programme.

    Avec `resume_dir`, on reprend un run existant: chaque étape déjà écrite sur disque est rechargée, seules les
    étapes manquantes sont refaites.
    """
    params = params or SolveParams()
    resuming = resume_dir is not None
    if resume_dir is None:
        run_dir = settings.runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "problem.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    else:
        run_dir = resume_dir
    console.print(f"[dim]{'Reprise du run' if resuming else 'Dossier du run'} : {run_dir}[/dim]")

    def saved(title: str, present: object) -> str:
        return f"{title} (repris du disque)" if present else title

    def save_research(r: Research) -> None:
        (run_dir / "research.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")

    research = _read_model(run_dir / "research.json", Research) if resuming else None
    stepper = Stepper(console, total=7 if (settings.web_enabled or research) else 6)

    if research is not None:
        stepper.next("Recherche web (reprise du disque)")
        if research.findings and research.brief.empty:  # arrêt survenu entre l'extraction et la synthèse
            try:
                research = await complete_brief(llm, settings, console, research)
                save_research(research)
            except Exception as e:  # noqa: BLE001 - on garde au moins les faits déjà extraits
                console.print(f"[yellow]Synthèse impossible ({type(e).__name__}: {e}). On continue sans dossier.[/yellow]")
    elif settings.web_enabled:
        stepper.next("Recherche web (Ctrl+C pour l'interrompre)")
        try:
            research = await interruptible(run_research(llm, settings, console, spec, on_progress=save_research))
        except Interrupted:
            console.print("\n[yellow]Recherche web interrompue.[/yellow]")
            if ask_continue is not None and not await ask_continue("Continuer la résolution sans le web ?"):
                raise UserQuit from None
        except Exception as e:  # noqa: BLE001 - réseau coupé ou fournisseur mal configuré: on continue sans le web
            console.print(f"[yellow]Recherche web indisponible ({type(e).__name__}: {e}). On continue sans.[/yellow]")
            log.debug("recherche web échouée", exc_info=True)

    analysis = _read_model(run_dir / "analysis.json", Analysis) if resuming else None
    stepper.next(saved("Analyse de la situation", analysis))
    if analysis is None:
        with console.status("Acteurs, incitations, leviers (modèle solide)..."):
            analysis = await analyze(llm, settings, spec, research.brief if research else None)
        (run_dir / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        log.debug("analyse: %d acteurs, %d leviers", len(analysis.actors), len(analysis.leverage_points))

    result = RunResult(run_dir=run_dir, spec=spec, analysis=analysis, research=research)

    raw = _read_lines(run_dir / "ideas_raw.jsonl", Idea, console) if resuming else None
    stepper.next(saved("Essaim d'idées", raw))
    if raw is None:
        raw = await generate_ideas(
            llm, settings, console, spec, analysis, params.ideas_target, params.per_call, result.brief
        )
        _dump(run_dir / "ideas_raw.jsonl", raw)
    result.raw_count = len(raw)

    unique = _read_lines(run_dir / "ideas_unique.jsonl", Idea, console) if resuming else None
    stepper.next(saved("Déduplication", unique))
    if unique is None:
        console.print(f"{len(raw)} idées comparées entre elles (embeddings)...")
        unique = await dedupe_ideas(llm, settings, raw)
        _dump(run_dir / "ideas_unique.jsonl", unique)
    result.unique_count = len(unique)

    audited = _read_lines(run_dir / "audits.jsonl", DeepResult, console) if resuming else None
    if audited is not None:
        stepper.next("Filtre et audit (repris du disque)")
        _restore_audit(result, unique, audited)
    else:
        await screen_and_audit(llm, settings, console, result, unique, params.pool, stepper)
    result.save()

    survivors = result.survivors
    if not survivors:
        console.print("[red]Aucune solution n'a survécu aux audits. Lance `encore`, ou ajoute des informations.[/red]")
        result.timings = stepper.finish()
        return result

    report_path = run_dir / "report.md"
    if resuming and report_path.exists():
        stepper.next("Rapport (repris du disque)")
    else:
        rejected = [f"{i.title}: {why}" for i, why in result.dropped] + [
            f"{r.idea.title}: {r.rejection}" for r in result.audit_rejects
        ]
        stepper.next("Rapport")
        with console.status("Rédaction du rapport (modèle solide)..."):
            report = await write_report(
                llm, settings, spec, analysis, ranked_for_report(result, params.top), rejected, result.stats, research
            )
        report_path.write_text(report, encoding="utf-8")
    result.timings = stepper.finish()
    return result
