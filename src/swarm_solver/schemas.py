from typing import Any

from pydantic import BaseModel, Field, model_validator

LENS_WEIGHTS = {"feasibility": 0.4, "constraints_risk": 0.35, "legality": 0.25}
# Part du score "impact attendu" (donné par le filtre) dans le classement final.
IMPACT_WEIGHT = 0.3
# Seuils des ALERTES affichées à l'utilisateur. Elles ne retirent aucune idée: c'est lui qui décide.
INFEASIBLE_MAX_SCORE = 3  # faisabilité notée à ce seuil ou moins
DOUBTFUL_LEGALITY_BELOW = 5  # légalité notée sous ce seuil
# Une idée jugée peu plausible par le filtre garde sa place, mais sa note est plafonnée à ceci.
IMPLAUSIBLE_SCORE_CAP = 4

# Seules exclusions codées en dur : on borne les MOYENS (illégal, nuisible), jamais l'objectif ni le "ton".
# Ajoutées à chaque prompt, indépendamment de ce que l'interview a produit.
BASELINE_RED_LINES = [
    "illegal acts: harassment, threats, defamation, fabricated or stolen evidence, spying on private accounts or communications",
    "provoking, staging or fabricating misconduct by a person",
    "physical violence, or a sustained campaign to psychologically torment someone",
]


class ProblemSpec(BaseModel):
    """Fiche problème produite par l'interview, base commune de tous les agents."""

    title: str
    stated_request: str = Field(description="Ce que l'utilisateur a demandé, avec ses mots")
    underlying_goal: str = Field(description="Le résultat voulu par l'utilisateur, conservé tel quel")
    context: str
    key_facts: list[str] = Field(default=[], description="Faits donnés par l'utilisateur, sans enjolivement")
    already_tried: list[str] = []
    constraints: list[str] = []
    resources: list[str] = []
    red_lines: list[str] = Field(default=[], description="Limites que l'utilisateur a lui-même posées")
    success_criteria: list[str] = []

    def to_prompt(self) -> str:
        def block(label: str, items: list[str]) -> str:
            return f"{label}:\n" + "\n".join(f"- {i}" for i in items) if items else f"{label}: (none)"

        return "\n".join(
            [
                f"PROBLEM: {self.title}",
                f"Stated request: {self.stated_request}",
                f"Goal: {self.underlying_goal}",
                f"Context: {self.context}",
                block("Key facts (true, do not contradict)", self.key_facts),
                block("Already tried", self.already_tried),
                block("Constraints", self.constraints),
                block("Available resources", self.resources),
                block("Red lines (never propose)", [*self.red_lines, *BASELINE_RED_LINES]),
                block("Success criteria", self.success_criteria),
            ]
        )


class Actor(BaseModel):
    name: str
    incentives: str = Field(description="Ce que cet acteur veut ou craint")
    leverage: str = Field(description="Ce qu'il peut faire, ou ce qu'on peut le pousser à faire")


class Analysis(BaseModel):
    """Lecture stratégique de la situation, partagée avec tout l'essaim."""

    # Champs clés obligatoires et non vides : sinon un petit modèle répond `[]` et l'essaim part sans analyse.
    adversarial: bool = Field(
        description="True si une personne ou une organisation s'oppose au but, l'entrave ou profite du problème"
    )
    root_causes: list[str] = Field(min_length=1)
    actors: list[Actor] = Field(min_length=1)
    why_previous_attempts_failed: list[str] = []
    leverage_points: list[str] = Field(min_length=1)
    assumptions_to_test: list[str] = []

    def to_prompt(self) -> str:
        def block(label: str, items: list[str]) -> str:
            return f"{label}:\n" + "\n".join(f"- {i}" for i in items) if items else ""

        actors = [f"{a.name}: wants/fears {a.incentives}; leverage: {a.leverage}" for a in self.actors]
        parts = [
            block("Root causes", self.root_causes),
            block("Actors", actors),
            block("Why previous attempts failed", self.why_previous_attempts_failed),
            block("Leverage points", self.leverage_points),
            block("Assumptions to test", self.assumptions_to_test),
        ]
        return "SITUATION ANALYSIS\n" + "\n".join(p for p in parts if p)


class InterviewStep(BaseModel):
    ready: bool = Field(description="True si on a assez d'informations pour résoudre")
    questions: list[str] = Field(default=[], max_length=3)


class IdeaDraft(BaseModel):
    title: str
    description: str
    steps: list[str] = Field(min_length=2, max_length=6, description="Premières actions concrètes")


class IdeaBatch(BaseModel):
    ideas: list[IdeaDraft]


class Idea(IdeaDraft):
    id: str
    lens: str

    def as_text(self) -> str:
        steps = "\n".join(f"  {n}. {s}" for n, s in enumerate(self.steps, 1))
        return f"{self.title}\n{self.description}\nSteps:\n{steps}"


class GateVerdict(BaseModel):
    """Filtre éliminatoire rapide, un appel par idée."""

    violates_red_line: bool
    illegal_or_harmful: bool
    plausible: bool
    score: int = Field(ge=0, le=10, description="Valeur attendue: effet sur l'objectif x probabilité que ça marche")
    reason: str

    @property
    def passed(self) -> bool:
        """Écarte seulement ce que l'utilisateur a interdit, ou ce qui repose sur un acte illégal ou nuisible.

        `plausible=False` n'écarte plus rien: c'est un jugement de petit modèle, la note est plafonnée à la place.
        """
        return not self.violates_red_line and not self.illegal_or_harmful


class LensVerdict(BaseModel):
    score: int = Field(ge=0, le=10)
    blocking: bool = Field(description="True si un défaut rend l'idée inutilisable")
    issues: list[str] = []
    mitigation: str = ""


class DeepResult(BaseModel):
    idea: Idea
    gate_score: int
    lenses: dict[str, LensVerdict]

    @model_validator(mode="before")
    @classmethod
    def _rename_old_lens(cls, data: Any) -> Any:
        """Les premiers runs nommaient la lentille de légalité `legal_ethics`: on la relit sous son nom actuel."""
        lenses = data.get("lenses") if isinstance(data, dict) else None
        if isinstance(lenses, dict) and "legal_ethics" in lenses and "legality" not in lenses:
            data = {**data, "lenses": {("legality" if k == "legal_ethics" else k): v for k, v in lenses.items()}}
        return data

    @property
    def rejection(self) -> str | None:
        """Raison du rejet par défaut, ou None. Seul l'illégal ou le gravement nuisible avéré écarte une idée.

        Ni le risque, ni la difficulté, ni une note de légalité basse n'écartent: ce sont des `warnings`. L'utilisateur
        peut de toute façon remettre une idée écartée dans le classement (`garder`).
        """
        legality = self.lenses["legality"]
        if legality.blocking:
            return "illégal ou nuisible: " + ("; ".join(legality.issues) or "jugé non recommandable")
        return None

    @property
    def warnings(self) -> list[str]:
        """Alertes à afficher à côté de l'idée. Elles informent, elles ne retirent rien."""
        out = []
        legality, feasibility = self.lenses["legality"], self.lenses["feasibility"]
        if legality.score < DOUBTFUL_LEGALITY_BELOW:
            out.append("légalité")
        if feasibility.blocking or feasibility.score <= INFEASIBLE_MAX_SCORE:
            out.append("faisabilité")
        if self.risk_level == "high":
            out.append("risque")
        return out

    @property
    def total(self) -> float:
        audits = sum(LENS_WEIGHTS[k] * v.score for k, v in self.lenses.items())
        return IMPACT_WEIGHT * self.gate_score + (1 - IMPACT_WEIGHT) * audits

    @property
    def risk_level(self) -> str:
        score = self.lenses["constraints_risk"].score
        return "low" if score >= 7 else "medium" if score >= 5 else "high"
