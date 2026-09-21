import logging
import math
import random
import secrets

from rich.console import Console

from .config import Settings
from .llm import LLMClient
from .logs import LOG
from .progress import gather_tolerant
from .research import ResearchBrief, brief_block
from .schemas import Analysis, Idea, IdeaBatch, ProblemSpec

log = logging.getLogger(f"{LOG}.ideation")

# Chaque worker reçoit un angle différent : c'est ce qui crée la diversité, bien plus que la température seule.
# Angles applicables à tout problème (technique, business, organisation, personnel...).
GENERAL_LENSES = [
    "the simplest thing that could work: cheapest and fastest actions, achievable within a week",
    "reframing: question whether this is the real problem; dissolve or sidestep it instead of solving it",
    "remove constraints: which limit (time, money, permission, information, skills) is the real bottleneck, and how to relax it",
    "small reversible experiments that produce evidence before committing",
    "inversion: list what would make things worse, then flip each into an action",
    "work backwards from the goal: critical path, sequencing and dependencies",
    "tools, automation, delegation or buying something that removes effort or uncertainty",
    "reduce scope or lower the bar: which smaller version of the goal captures most of the value",
    "analogies borrowed from very different fields (medicine, logistics, games, ecology, law...)",
    "leveraging people: allies, mentors, intermediaries, communities, unlikely partners",
    "hedging: options that stay good whether or not your key assumption turns out to be true",
    "use existing institutions, rules, standards or outside bodies to get leverage",
    "radical, unconventional approaches that break an assumption of the problem",
    "communication and negotiation tactics, including framing, timing and who says it",
    "long-term positioning that makes the problem irrelevant in 6-12 months",
]
# Ajoutés seulement si l'analyse détecte que quelqu'un s'oppose au but, l'entrave ou profite du problème.
ADVERSARIAL_LENSES = [
    "power and incentives: who benefits from the status quo, and how to change their calculation",
    "hardball but lawful tactics: pressure, timing, visibility, making inaction costly for those who benefit from the problem",
    "make the real problem objectively visible: evidence, metrics, records, audits by third parties",
    "reduce the opponent's leverage without a direct confrontation: scope, information flow, dependencies",
    "exit, exemption or repositioning options: move, swap, negotiate a departure, change the game",
    "escalation sequencing: what to do first, second, third, and when to go over someone's head",
]


def select_lenses(analysis: Analysis) -> list[str]:
    """Angles généraux, entrecoupés d'angles "rapport de force" seulement si quelqu'un s'oppose à l'objectif.

    Entrecoupés pour que les petits runs (peu d'appels) voient les deux familles dès les premiers lots.
    """
    if not analysis.adversarial:
        return list(GENERAL_LENSES)
    mixed: list[str] = []
    for i in range(max(len(GENERAL_LENSES), len(ADVERSARIAL_LENSES))):
        mixed += [seq[i] for seq in (GENERAL_LENSES, ADVERSARIAL_LENSES) if i < len(seq)]
    return mixed


RULES = """Rules:
- Ideas may be uncomfortable, ruthless, cynical, manipulative, ethically questionable, politically incorrect or
  socially awkward. Do not soften, moralize or self-censor: the reader decides. Ethics and tone are not criteria.
- The ONLY hard exclusions are unlawful means: illegal acts, harassment, threats, defamation (false statements
  that damage someone), fabricated or stolen evidence, spying on private accounts, provoking or staging
  misconduct by someone, physical violence, a sustained campaign to torment someone. Everything lawful is allowed,
  including pressure, hardball politics and using true facts against someone. Legal system of reference:
  {jurisdiction}.
- Each idea: a short title, a 2-3 sentence description (who does what, and why it works given the actors'
  incentives), and 2-5 concrete first steps.
- The most obvious solution (the textbook answer anyone would think of first) is covered by other members of the
  swarm: do NOT propose it. Every idea you give must
  work through a DIFFERENT mechanism from the others.
- Be specific to THIS situation and use the situation analysis. No generic advice ("communicate better",
  "raise awareness", "be more organised").
- Ideas may need help, resources, allies or an authority the user does not have yet: then say what is needed and
  how to get it. Do not limit yourself to what the user could do alone tomorrow. If someone is an obstacle, do not
  count on their goodwill unless the idea explains why they would cooperate.
- If a web research brief is present, use its facts (procedures, deadlines, what worked and what backfired for
  others) and prefer tactics proven in real cases. Do not propose what it reports as backfiring.
- Respect the key facts, the constraints and the red lines.
- Write in {language}."""

WORKER = (
    """You are one member of an idea-generation swarm working for a person who wants a result and needs real options,
not lectures. Produce {n} DISTINCT solutions, using ONLY this lens:

LENS: {lens}

"""
    + RULES
)

VARIANTS = (
    """You are one member of a swarm exploring VARIATIONS of one promising solution. Produce {n} DISTINCT variants or
follow-ups of the BASE solution: change the mechanism, the target, the timing, the intensity, the cost, the
messenger, or combine it with another approach. {direction}

"""
    + RULES
)


def _user_prompt(spec: ProblemSpec, analysis: Analysis, brief: ResearchBrief | None, extra: str = "") -> str:
    return f"{spec.to_prompt()}\n\n{analysis.to_prompt()}{brief_block(brief)}{extra}"


async def generate_ideas(
    llm: LLMClient,
    settings: Settings,
    console: Console,
    spec: ProblemSpec,
    analysis: Analysis,
    target: int,
    per_call: int = 10,
    brief: ResearchBrief | None = None,
) -> list[Idea]:
    rng = random.Random()
    calls = math.ceil(target / per_call)
    lenses = select_lenses(analysis)
    jobs = []
    for i in range(calls):
        lens = lenses[i % len(lenses)]
        system = WORKER.format(
            n=per_call, lens=lens, language=settings.language, jurisdiction=settings.jurisdiction
        )
        jobs.append(
            _batch(llm, settings, system, _user_prompt(spec, analysis, brief), lens, rng.uniform(0.7, 1.1),
                   rng.randrange(2**31), tag=f"essaim {i + 1}/{calls}")
        )
    return await _collect(console, "Essaim d'idées", jobs, prefix="I")


async def explore_variants(
    llm: LLMClient,
    settings: Settings,
    console: Console,
    spec: ProblemSpec,
    analysis: Analysis,
    base: Idea,
    direction: str,
    target: int,
    per_call: int = 10,
    brief: ResearchBrief | None = None,
) -> list[Idea]:
    """Génère des variantes d'une solution existante, éventuellement dans une direction demandée."""
    rng = random.Random()
    calls = math.ceil(target / per_call)
    direction_text = f"USER DIRECTION: {direction}" if direction else "Cover several different directions."
    system = VARIANTS.format(
        n=per_call, direction=direction_text, language=settings.language, jurisdiction=settings.jurisdiction
    )
    user = _user_prompt(spec, analysis, brief, f"\n\nBASE SOLUTION:\n{base.as_text()}")
    lens = f"variante de {base.id}" + (f" ({direction})" if direction else "")
    jobs = [
        _batch(llm, settings, system, user, lens, rng.uniform(0.8, 1.1), rng.randrange(2**31),
               tag=f"variantes {base.id} {i + 1}/{calls}")
        for i in range(calls)
    ]
    # Suffixe aléatoire : deux appels `variantes` sur la même base ne doivent pas produire les mêmes identifiants.
    return await _collect(console, "Variantes", jobs, prefix=f"{base.id}.{secrets.token_hex(2)}-")


async def _collect(console: Console, label: str, jobs: list, prefix: str) -> list[Idea]:
    batches, failures, err = await gather_tolerant(console, label, jobs)
    if failures:
        console.print(f"[yellow]{failures}/{len(jobs)} appels ont échoué (dernier: {err})[/yellow]")
    ideas: list[Idea] = []
    for lens, batch in batches:
        for draft in batch.ideas:
            ideas.append(Idea(id=f"{prefix}{len(ideas):05d}", lens=lens, **draft.model_dump()))
    log.debug("%s: %d idées brutes", label, len(ideas))
    return ideas


async def _batch(
    llm: LLMClient, settings: Settings, system: str, user: str, lens: str, temperature: float, seed: int, tag: str
) -> tuple[str, IdeaBatch]:
    batch = await llm.structured(
        settings.fast_model, system, user, IdeaBatch, temperature=temperature, seed=seed, tag=tag
    )
    return lens, batch
