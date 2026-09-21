import numpy as np
from rich.console import Console

from swarm_solver.config import Settings
from swarm_solver.dedup import pick_diverse, select_diverse
from swarm_solver.schemas import GateVerdict, Idea
from swarm_solver.verify import gate

from test_pipeline import SPEC


def verdict(score: int) -> GateVerdict:
    return GateVerdict(violates_red_line=False, illegal_or_harmful=False, plausible=True, score=score, reason="ok")


def idea(n: int, title: str) -> Idea:
    return Idea(id=f"I{n}", lens="l", title=title, description="d", steps=["a", "b"])


A, A_BIS, B, C = [1.0, 0.0, 0.0], [0.99, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


# --- select_diverse ------------------------------------------------------------------------------------------


def test_a_near_duplicate_is_skipped_in_favour_of_a_lower_scored_different_idea():
    vectors = np.array([A, A_BIS, B], dtype=np.float32)
    assert select_diverse(vectors, [9, 9, 6], k=2) == [0, 2]  # A puis B, pas A puis sa copie


def test_without_diversity_it_is_a_plain_ranking_by_score():
    vectors = np.array([A, A_BIS, B], dtype=np.float32)
    assert select_diverse(vectors, [9, 9, 6], k=2, diversity=0) == [0, 1]


def test_ideas_similar_to_already_audited_ones_are_avoided():
    vectors = np.array([A, B], dtype=np.float32)
    existing = np.array([A], dtype=np.float32)  # une idée quasi identique à la première est déjà auditée
    assert select_diverse(vectors, [9, 6], k=1, existing=existing) == [1]


def test_a_clearly_better_idea_still_wins_over_a_slightly_different_one():
    """La variété départage, elle ne remplace pas la note: 9 contre 2 reste 9, même un peu ressemblant."""
    vectors = np.array([A, [0.6, 0.8, 0.0]], dtype=np.float32)
    assert select_diverse(vectors, [2, 9], k=1) == [1]


def test_select_diverse_handles_empty_and_oversized_requests():
    assert select_diverse(np.empty((0, 3), dtype=np.float32), [], k=5) == []
    vectors = np.array([A, B], dtype=np.float32)
    assert sorted(select_diverse(vectors, [5, 5], k=10)) == [0, 1]


# --- pick_diverse avec un faux modèle d'embeddings --------------------------------------------------------------


class TitleEmbedder:
    """Les idées dont le titre commence par "dup" partagent le même vecteur; les autres ont chacune le leur."""

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, 0
        self._own: dict[str, int] = {}

    async def embed(self, model, texts):
        self.calls += 1
        if self.fail:
            raise ConnectionError("modèle d'embeddings absent")
        out = np.zeros((len(texts), 16), dtype=np.float32)
        for row, text in enumerate(texts):
            key = 0 if text.startswith("dup") else self._own.setdefault(text, len(self._own) + 1)
            out[row, key] = 1.0
        return out


async def test_pick_diverse_takes_one_of_the_copies_and_the_distinct_ideas():
    cands = [(idea(1, "dup a"), verdict(9)), (idea(2, "dup b"), verdict(9)), (idea(3, "dup c"), verdict(9)),
             (idea(4, "autre 1"), verdict(7)), (idea(5, "autre 2"), verdict(6))]
    picked = await pick_diverse(TitleEmbedder(), Settings(), cands, k=3)
    titles = [i.title for i, _ in picked]
    assert len([t for t in titles if t.startswith("dup")]) == 1 and {"autre 1", "autre 2"} <= set(titles)


async def test_pick_diverse_falls_back_to_the_score_ranking_if_embeddings_fail():
    cands = [(idea(1, "dup a"), verdict(9)), (idea(2, "dup b"), verdict(9)), (idea(3, "autre"), verdict(7))]
    picked = await pick_diverse(TitleEmbedder(fail=True), Settings(), cands, k=2)
    assert [i.title for i, _ in picked] == ["dup a", "dup b"]  # pas de plantage: repli sur la note


async def test_pick_diverse_skips_embeddings_when_everything_fits_or_diversity_is_off():
    llm = TitleEmbedder()
    cands = [(idea(1, "a"), verdict(5)), (idea(2, "b"), verdict(8))]
    assert [i.id for i, _ in await pick_diverse(llm, Settings(), cands, k=5)] == ["I2", "I1"]
    many = cands + [(idea(3, "c"), verdict(1))]
    await pick_diverse(llm, Settings(diversity=0), many, k=2)
    assert llm.calls == 0


# --- dans le filtre -------------------------------------------------------------------------------------------


class GateLLM(TitleEmbedder):
    """Le filtre note "dup*" à 9 (la même idée reformulée 4 fois), les autres à 7 et 6."""

    async def structured(self, model, system, user, schema, **kwargs):
        title = user.split("PROPOSED SOLUTION:\n", 1)[1].split("\n", 1)[0]
        return verdict(9 if title.startswith("dup") else 7 if title == "autre 1" else 6)


async def test_the_gate_sends_a_varied_batch_to_the_audit_and_keeps_the_rest_in_reserve():
    ideas = [idea(n, f"dup {n}") for n in range(4)] + [idea(10, "autre 1"), idea(11, "autre 2")]
    outcome = await gate(GateLLM(), Settings(), Console(quiet=True), SPEC, ideas, keep=3)

    kept = [i.title for i, _ in outcome.kept]
    assert len([t for t in kept if t.startswith("dup")]) == 1 and "autre 1" in kept and "autre 2" in kept
    assert {i.title for i, _ in outcome.reserve} == {"dup 1", "dup 2", "dup 3"}  # les copies attendent en réserve
    assert len(outcome.kept) + len(outcome.reserve) == 6  # rien ne se perd
