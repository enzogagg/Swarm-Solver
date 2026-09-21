import asyncio
import logging
from dataclasses import dataclass, field

from rich.console import Console

from .config import Settings
from .dedup import pick_diverse
from .llm import LLMClient
from .logs import LOG
from .progress import gather_tolerant
from .research import ResearchBrief, brief_block
from .schemas import IMPLAUSIBLE_SCORE_CAP, LENS_WEIGHTS, DeepResult, GateVerdict, Idea, LensVerdict, ProblemSpec

log = logging.getLogger(f"{LOG}.verify")

GATE = """You screen ONE proposed solution for a person who wants a result and needs real options.

Tone, taste, ethics and social acceptability are NOT criteria. Ruthless, cynical, manipulative, ethically
questionable, politically incorrect or risky ideas are all acceptable as long as they are lawful ({jurisdiction} law):
pressure, strategic silence, exploiting people's incentives, playing organisational politics, using true facts to make
someone look bad, and so on.

- illegal_or_harmful=true ONLY if the idea requires: an illegal act, harassment, threats, defamation (false
  statements that damage someone), fabricated or stolen evidence, spying on private accounts or communications,
  provoking or staging misconduct by someone, physical violence, or a sustained campaign to torment someone.
  Being ethically questionable is NOT a reason to set it.
- violates_red_line=true only if it crosses a limit the user explicitly stated.
- plausible=false only if the idea is too vague to act on. A step that needs an authority, access or status the user
  does not have today (read their position in the key facts and constraints), or that depends on people who obstruct
  the goal acting against their own interest, is NOT a reason to discard: it is a reason for a LOWER score. The user
  decides what is worth pursuing, so ideas that need help or a bold move still deserve a fair score.
- score 0-10 = expected value: how much it moves the user's goal, times how likely it is to work given the actual
  power relations and constraints. Anchors: 8-10 = forces someone to act even if those who obstruct stay passive, and
  the user can start it alone this week; 6-7 = credible but depends on someone's goodwill or on a scarce ally;
  3-5 = long shot or mostly symbolic; 0-2 = generic advice that fits any situation ("communicate better",
  "plan better") or cannot work. Use the whole range.
- reason: one sentence naming the decisive factor: who has to act, and why they would.
Write in {language}."""

LENS_PROMPTS = {
    "feasibility": """You audit EXECUTION REALISM of one solution: could this person carry it out from where they
stand, given the key facts, constraints and the resources they DO have? Look for missing prerequisites, unrealistic
timelines, and steps that depend on an actor who is stated to be hostile to the goal or to benefit from the
problem, with no workaround. If the resources list is empty, judge whether an ordinary person in their position could do the steps;
do not assume the worst. Difficulty, or dependence on someone's cooperation, is a RISK: lower the score. Set
blocking=true only if it looks impossible from where the user stands today; the user will still see the idea and
decide, so state precisely what would have to change to make it possible (in issues or mitigation).""",
    "constraints_risk": """You audit CONSTRAINTS and RISK of one solution: does it respect every stated constraint,
and what are the realistic downsides and failure modes for the user (career, relationships, reputation, money,
time, retaliation, backfire)? A high score means low exposure. Report the risks plainly; risk alone is NEVER a
reason to block. blocking=true only if a stated constraint is directly violated.""",
    "legality": """You audit LEGALITY and REAL HARM of one solution under {jurisdiction} law: is it illegal (harassment,
defamation, threats, fabricated or stolen evidence, privacy violation, unlawful recording or access, provoking or
staging misconduct, physical violence, a sustained campaign of torment) or does it seriously harm an innocent third
party? Rudeness, hardball, ruthlessness, manipulation as persuasion or pressure, and anything ethically questionable
or politically incorrect but lawful are NOT problems. blocking=true only for something illegal or seriously harmful.
If the idea is lawful but aggressive or ethically gray, say so in issues as "risk: ..." and do not block.
score below 5 means it should not be recommended.""",
}

LENS_SUFFIX = """
If a web research brief is provided, use it: check the idea against the procedures, deadlines and legal
rules it reports, and against what it says backfired for others.
score: 0-10 (be honest: 5 is mediocre). issues: up to 3 concrete problems. mitigation: one sentence that would
reduce the main problem, or empty. Write in {language}."""


@dataclass
class GateOutcome:
    kept: list[tuple[Idea, GateVerdict]] = field(default_factory=list)
    reserve: list[tuple[Idea, GateVerdict]] = field(default_factory=list)  # ont passé le filtre, pas encore audités
    dropped: list[tuple[Idea, GateVerdict]] = field(default_factory=list)


async def gate(
    llm: LLMClient,
    settings: Settings,
    console: Console,
    spec: ProblemSpec,
    ideas: list[Idea],
    keep: int,
    already_audited: list[Idea] | None = None,
) -> GateOutcome:
    """Filtre éliminatoire sur toutes les idées avec le modèle rapide.

    Envoie à l'audit `keep` idées bien notées ET variées (pas quatre versions de la même idée), loin de celles déjà
    auditées; les autres passent en réserve, ordonnées par note.
    """
    system = GATE.format(language=settings.language, jurisdiction=settings.jurisdiction)

    async def one(idea: Idea) -> tuple[Idea, GateVerdict]:
        verdict = await llm.structured(
            settings.fast_model, system, f"{spec.to_prompt()}\n\nPROPOSED SOLUTION:\n{idea.as_text()}", GateVerdict,
            temperature=0.1, tag=f"filtre {idea.id}",
        )
        log.debug(
            "filtre %s %r -> %s, score %d: %s",
            idea.id, idea.title, "OK" if verdict.passed else "ÉCARTÉE", verdict.score, verdict.reason,
        )
        return idea, verdict

    results, failures, err = await gather_tolerant(console, "Filtre éliminatoire", [one(i) for i in ideas])
    if failures:
        console.print(f"[yellow]{failures} idées non évaluées (dernier: {err})[/yellow]")

    for idea, verdict in results:
        if not verdict.plausible and verdict.score > IMPLAUSIBLE_SCORE_CAP:
            log.debug("filtre %s jugée peu plausible: note %d plafonnée à %d", idea.id, verdict.score, IMPLAUSIBLE_SCORE_CAP)
            verdict.score = IMPLAUSIBLE_SCORE_CAP
    passed = sorted(((i, v) for i, v in results if v.passed), key=lambda iv: iv[1].score, reverse=True)
    kept = await pick_diverse(llm, settings, passed, keep, already_audited)
    kept_ids = {i.id for i, _ in kept}
    return GateOutcome(
        kept=kept,
        reserve=[iv for iv in passed if iv[0].id not in kept_ids],
        dropped=[(i, v) for i, v in results if not v.passed],
    )


async def deep_verify(
    llm: LLMClient,
    settings: Settings,
    console: Console,
    spec: ProblemSpec,
    shortlist: list[tuple[Idea, GateVerdict]],
    brief: ResearchBrief | None = None,
) -> list[DeepResult]:
    """Trois vérificateurs indépendants (faisabilité, contraintes/risques, légalité) par idée."""

    async def audit(idea: Idea, lens: str) -> tuple[str, LensVerdict]:
        verdict = await llm.structured(
            settings.strong_model,
            LENS_PROMPTS[lens].format(jurisdiction=settings.jurisdiction)
            + LENS_SUFFIX.format(language=settings.language),
            f"{spec.to_prompt()}{brief_block(brief)}\n\nPROPOSED SOLUTION:\n{idea.as_text()}",
            LensVerdict,
            temperature=0.1,
            tag=f"audit {lens} {idea.id}",
        )
        return lens, verdict

    async def one(idea: Idea, gate_score: int) -> DeepResult:
        lenses = dict(await asyncio.gather(*(audit(idea, lens) for lens in LENS_WEIGHTS)))
        result = DeepResult(idea=idea, gate_score=gate_score, lenses=lenses)
        log.debug(
            "audit %s %r -> total %.1f, risque %s, %s",
            idea.id, idea.title, result.total, result.risk_level, result.rejection or "retenue",
        )
        return result

    results, failures, err = await gather_tolerant(
        console, "Vérification approfondie", [one(i, v.score) for i, v in shortlist]
    )
    if failures:
        console.print(f"[yellow]{failures} idées non vérifiées (dernier: {err})[/yellow]")
    return results
