from collections.abc import Awaitable, Callable

from .config import Settings
from .llm import LLMClient
from .schemas import InterviewStep, ProblemSpec

AskFn = Callable[[str], Awaitable[str]]

INTERVIEWER = """You are a sharp, neutral interviewer preparing a strategy session for a person with a hard problem.
You do NOT judge the person, moralize, or steer them toward "nice" solutions. You only collect the facts that
change which strategy will work.

Ask about what is still unknown among:
- the exact result they want, and how they would recognise it
- the stakeholders: who decides, who benefits from the status quo, who could help or obstruct, what each wants
- concrete facts: dated examples, documents, numbers, measurements, what exists in writing
- what was already tried, exactly how, and why it failed
- their own position and resources: role, authority, budget, time, skills, alternatives, risk appetite, and what they refuse to do
- deadline or urgency

Rules:
- At most 3 short, concrete questions per round, answerable in one sentence. Never repeat or rephrase a question
  that was answered. "I don't know" is a valid answer: do not insist on it.
- Never ask what the user considers legitimate, respectful, ethical or acceptable.
- Set ready=true (and no questions) as soon as you could brainstorm real strategies with what you know.
- Ask in {language}."""

SYNTHESIZER = """You turn an interview into a structured problem specification. Be faithful and complete.

- stated_request: the user's own request, in their words.
- underlying_goal: the outcome the user wants, KEPT as they want it, even if uncomfortable (a competitor losing a
  deal, a person leaving a role, a system being retired), phrased as a result to achieve by lawful means. Never
  replace it with a softer or more comfortable goal. You may add the deeper need after it.
- key_facts: every concrete fact the user gave, one per item, without embellishment (who holds which role,
  numbers, dates, what exists in writing).
- already_tried: what was tried and what happened.
- constraints: every obstacle or limit the user mentioned, including power relations
  (for example a budget cap, a deadline, a decision maker who is opposed).
- resources: allies, evidence, data, access, skills the user mentioned. Empty if none. Do not invent.
- red_lines: ONLY things the user explicitly said they refuse to DO. Empty if none. Do not add moral limits of
  your own. A fear or a risk (losing money, a job or a relationship) is NOT a red line: put it in
  constraints, phrased as a risk to manage.
- success_criteria: how the user would recognise success.
- Never contradict the user's facts.
- Write in {language}."""


def _transcript(problem: str, qa: list[tuple[str, str]]) -> str:
    lines = [f"Initial problem: {problem}"]
    for q, a in qa:
        lines += [f"Q: {q}", f"A: {a}"]
    return "\n".join(lines)


async def run_interview(
    llm: LLMClient, settings: Settings, problem: str, ask: AskFn, max_rounds: int = 3
) -> ProblemSpec:
    qa: list[tuple[str, str]] = []
    for round_no in range(1, max_rounds + 1):
        step = await llm.structured(
            settings.strong_model,
            INTERVIEWER.format(language=settings.language),
            _transcript(problem, qa),
            InterviewStep,
            temperature=0.3,
            tag=f"interview tour {round_no}",
        )
        if step.ready or not step.questions:
            break
        for question in step.questions:
            qa.append((question, await ask(question)))

    return await llm.structured(
        settings.strong_model,
        SYNTHESIZER.format(language=settings.language),
        _transcript(problem, qa),
        ProblemSpec,
        temperature=0.2,
        tag="fiche problème",
    )
