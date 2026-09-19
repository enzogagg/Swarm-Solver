from .config import Settings
from .llm import LLMClient
from .research import ResearchBrief, brief_block
from .schemas import Analysis, ProblemSpec

ANALYST = """You are a candid strategist. Analyse the situation like a chess player, not a counsellor: who has power,
what each actor wants or fears, why the current equilibrium holds, and where it can be moved.

- adversarial: true if a person or organisation opposes the goal, obstructs it, or profits from the problem
  persisting; false for purely technical, financial or planning problems.
- root_causes: why the problem exists and persists.
- actors: every party that matters (the user, anyone opposing or obstructing the goal, anyone who benefits from the
  current situation, decision makers, bystanders, outside bodies), with their incentives and their leverage. Include
  what each one gains from the current situation. For a purely technical problem, list the systems, teams or
  constraints involved.
- why_previous_attempts_failed: specific reasons, from the facts.
- leverage_points: concrete places where a small lawful action could shift the balance.
- assumptions_to_test: important things you are guessing and that the user should verify.

If a web research brief is provided, rely on it for the legal and procedural facts and for how similar cases
played out, and do not contradict it.
No moralizing, no generic advice. Use only the facts provided and mark guesses as assumptions. Write in {language}."""


async def analyze(
    llm: LLMClient, settings: Settings, spec: ProblemSpec, brief: ResearchBrief | None = None
) -> Analysis:
    return await llm.structured(
        settings.strong_model,
        ANALYST.format(language=settings.language),
        spec.to_prompt() + brief_block(brief),
        Analysis,
        temperature=0.3,
        tag="analyse de la situation",
    )
