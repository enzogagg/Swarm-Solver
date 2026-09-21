from .config import Settings
from .llm import LLMClient
from .research import Research, brief_block
from .schemas import Analysis, DeepResult, ProblemSpec

REPORTER = """You write the decision report for a person facing a hard problem. Be direct, candid and useful.
No moralizing, no disclaimers, and do not soften ideas that are lawful: state their risks plainly instead.

Structure (Markdown, in {language}):
1. "Le problème en clair": 3 sentences that keep the user's real goal and facts. Never contradict the KEY FACTS.
2. "Lecture de la situation": 4-6 bullets from the SITUATION ANALYSIS: who holds power, why the status quo holds,
   where the leverage is.
3. "Recommandation": the best solution, why it fits the power relations, and a plan for the first 7 days.
4. "Alternatives": the next best solutions, one short paragraph each, and when to prefer them.
5. "Coups audacieux": solutions marked risk=high. For each: what it could win, what it could cost, and under which
   conditions to consider it. Omit this section if there are none.
6. "Ce qui a été écarté": one paragraph on what was removed and why (see REJECTED), and remind that the user can
   bring any of them back with the command `garder`.

Solutions may carry FLAGS (legality, feasibility, risk). They are warnings, not verdicts: the user decides. State
each flag plainly next to its solution, and never hide or drop a flagged solution.
Refer to solutions by their number, e.g. "#3". When a statement relies on the WEB RESEARCH BRIEF, cite its source ids
like [S3]; never invent an id. Use only the material provided; do not invent facts."""


def _render(ranked: list[tuple[int, DeepResult]]) -> str:
    out = []
    for rank, r in ranked:
        audits = "\n".join(
            f"  - {lens}: {v.score}/10; issues: {'; '.join(v.issues) or 'none'}; mitigation: {v.mitigation or 'n/a'}"
            for lens, v in r.lenses.items()
        )
        out.append(
            f"#{rank} (score {r.total:.1f}, risk={r.risk_level}, flags={', '.join(r.warnings) or 'none'}) [{r.idea.lens}]\n{r.idea.as_text()}\nAudits:\n{audits}"
        )
    return "\n\n".join(out)


async def write_report(
    llm: LLMClient,
    settings: Settings,
    spec: ProblemSpec,
    analysis: Analysis,
    ranked: list[tuple[int, DeepResult]],
    rejected: list[str],
    stats: dict[str, int],
    research: Research | None = None,
) -> str:
    funnel = " -> ".join(f"{k}: {v}" for k, v in stats.items())
    rejected_text = "\n".join(f"- {r}" for r in rejected[:15]) or "(nothing notable)"
    user = (
        f"{spec.to_prompt()}\n\n{analysis.to_prompt()}{brief_block(research.brief if research else None)}\n\nFUNNEL: {funnel}\n\n"
        f"SOLUTIONS:\n{_render(ranked)}\n\nREJECTED (sample):\n{rejected_text}"
    )
    body = await llm.text(settings.strong_model, REPORTER.format(language=settings.language), user, tag="rapport")
    sources = research.sources_markdown(body) if research else ""
    tail = f"\n## Sources\n\n{sources}\n" if sources else ""
    return f"# {spec.title}\n\n> Entonnoir: {funnel}\n\n{body}\n{tail}"
