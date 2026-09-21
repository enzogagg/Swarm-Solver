import logging
from collections.abc import Awaitable, Callable

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from .config import Settings
from .dedup import dedupe_ideas, pick_diverse
from .ideation import explore_variants
from .interrupts import Interrupted, UserQuit, interruptible
from .llm import LLMClient
from .logs import LOG
from .pipeline import RunResult, SolveParams, screen_and_audit, solve
from .research import ask_web, brief_block
from .pipeline import RejectedItem
from .schemas import DeepResult, GateVerdict
from .ui import score_bar
from .verify import deep_verify

log = logging.getLogger(f"{LOG}.session")

ReadFn = Callable[[str], Awaitable[str]]

HELP = """[bold]Commandes[/bold]
  [cyan]N[/cyan]                     détail complet de la solution N (plan jour par jour, messages, plan B)
  [cyan]variantes N \\[consigne][/cyan]  génère et audite des variantes de la solution N, ex: variantes 2 plus discret
  [cyan]encore[/cyan]                audite le lot suivant d'idées gardées en réserve
  [cyan]liste[/cyan]                 réaffiche le classement
  [cyan]rejetées[/cyan]              ce qui n'est pas dans le classement, avec une référence (R1, R2...) et la raison
  [cyan]garder R2[/cyan]             remet une idée écartée dans le classement: c'est toi qui décides, pas le programme
  [cyan]écarter 3[/cyan]             retire la solution 3 du classement (elle reste dans `rejetées`)
  [cyan]info <texte>[/cyan]          ajoute un fait à la fiche et relance toute la résolution
  [cyan]recherche <question>[/cyan]  cherche sur le web pour répondre à une question précise, avec sources
  [cyan]sources[/cyan]               liste les sources web collectées
  [cyan]<question libre>[/cyan]      discute des solutions, ex: et si le budget est divisé par deux ?
  [cyan]q[/cyan]                     quitter
  [dim]Ctrl+C annule la commande en cours et te ramène ici.[/dim]"""

DETAILER = """You are a strategist turning ONE chosen solution into an execution plan for the user. Be concrete and candid.

Give, in Markdown and in {language}:
1. Preconditions: what to check or obtain first.
2. A dated plan: day by day for week 1, then week by week.
3. Ready-to-use material where relevant: exact wording of key messages, commands, checklists or templates.
4. Who to involve, and what each of them wants.
5. Signals that it is working, signals that it is failing, and when to pivot.
6. The main risk and how to limit it.

No moralizing. If a step would be unlawful ({jurisdiction} law), replace it with a lawful way to the same result and
say so in one line, so the user knows what changed and why.
Use only the facts provided."""

ADVISOR = """You are the user's strategy advisor. You know their situation and the ranked solutions below.
Answer their last message directly and concretely, in {language}. Reason about power and incentives.
No moralizing. Refer to solutions by number. If asked to help with something unlawful or harmful to someone
(fabricated evidence, harassment, entrapment, spying...), say in one sentence that you will not help with that
part and give the lawful route to the same result."""

HISTORY_TURNS = 6


class Session:
    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        console: Console,
        result: RunResult,
        params: SolveParams,
        read: ReadFn,
    ):
        self.llm, self.settings, self.console = llm, settings, console
        self.result, self.params, self.read = result, params, read
        self.history: list[tuple[str, str]] = []

    # --- affichage ---------------------------------------------------------------------------------------------

    def show_table(self) -> None:
        table = Table(title="Solutions retenues", box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
        table.add_column("#", justify="right", style="bold")
        table.add_column("Score", no_wrap=True)
        table.add_column("Risque", no_wrap=True)
        table.add_column("Alertes", style="yellow", overflow="fold")
        table.add_column("Solution", overflow="fold")
        table.add_column("Angle", style="dim", overflow="fold")
        colors = {"low": "green", "medium": "yellow", "high": "red"}
        for rank, r in enumerate(self.result.survivors, 1):
            table.add_row(
                str(rank), score_bar(r.total), f"[{colors[r.risk_level]}]{r.risk_level}[/]",
                ("⚠ " + ", ".join(r.warnings)) if r.warnings else "",
                escape(r.idea.title), escape(r.idea.lens[:48]),
            )
        self.console.print(table)
        s = self.result.stats
        self.console.print(f"[dim]{s['retenues']} retenues, {s['en réserve']} en réserve (`encore`), {s['écartées par le filtre']} écartées[/dim]")

    def _pick(self, arg: str) -> DeepResult | None:
        survivors = self.result.survivors
        if not arg.strip().isdigit() or not 1 <= int(arg) <= len(survivors):
            self.console.print(f"[red]Numéro invalide: choisis entre 1 et {len(survivors)}.[/red]")
            return None
        return survivors[int(arg) - 1]

    def _context(self, limit: int = 12) -> str:
        lines = [
            f"#{rank} [risk={r.risk_level}] {r.idea.title}: {r.idea.description}"
            for rank, r in enumerate(self.result.survivors[:limit], 1)
        ]
        return "\n".join(lines) or "(none yet)"

    # --- commandes ---------------------------------------------------------------------------------------------

    async def run(self) -> None:
        self.show_table()
        self.console.print(HELP)
        while True:
            try:
                line = (await self.read("\nswarm> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in {"q", "quit", "exit", "quitter"}:
                break
            try:
                await interruptible(self._dispatch(line))
            except Interrupted:
                self.console.print("\n[yellow]Commande interrompue.[/yellow]")
            except UserQuit:
                self.console.print("[yellow]Commande annulée.[/yellow]")
            except Exception as e:  # noqa: BLE001 - une commande ratée ne doit pas fermer la session
                log.debug("commande %r échouée", line, exc_info=True)
                self.console.print(f"[red]Échec: {type(e).__name__}: {e}[/red]")

    async def _dispatch(self, line: str) -> None:
        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()
        if line.isdigit():
            await self._detail(line)
        elif cmd in {"variantes", "variante", "v"}:
            n, _, direction = rest.strip().partition(" ")
            await self._variants(n, direction.strip())
        elif cmd in {"encore", "more"}:
            await self._more()
        elif cmd in {"liste", "list", "l"}:
            self.show_table()
        elif cmd in {"rejetées", "rejetees", "rejected"}:
            self._rejected()
        elif cmd in {"garder", "keep"} and rest.strip():
            await self._keep(rest.strip())
        elif cmd in {"écarter", "ecarter", "retirer"} and rest.strip():
            self._veto(rest.strip())
        elif cmd == "info" and rest.strip():
            await self._info(rest.strip())
        elif cmd in {"recherche", "web"} and rest.strip():
            await self._web(rest.strip())
        elif cmd == "sources":
            self._sources()
        elif cmd in {"aide", "help", "?"}:
            self.console.print(HELP)
        else:
            await self._question(line)

    async def _detail(self, arg: str) -> None:
        r = self._pick(arg)
        if not r:
            return
        system = DETAILER.format(language=self.settings.language, jurisdiction=self.settings.jurisdiction)
        audits = "\n".join(f"- {k}: {v.score}/10, {'; '.join(v.issues)}" for k, v in r.lenses.items())
        user = f"{self.result.spec.to_prompt()}\n\n{self.result.analysis.to_prompt()}{brief_block(self.result.brief)}\n\nCHOSEN SOLUTION:\n{r.idea.as_text()}\n\nAUDITS:\n{audits}"
        with self.console.status("Rédaction du plan..."):
            plan = await self.llm.text(self.settings.strong_model, system, user, tag=f"détail {r.idea.id}")
        (self.result.run_dir / f"detail-{r.idea.id}.md").write_text(f"# {r.idea.title}\n\n{plan}\n", encoding="utf-8")
        self.console.print(Markdown(f"# {r.idea.title}\n\n{plan}"))

    async def _variants(self, arg: str, direction: str) -> None:
        base = self._pick(arg)
        if not base:
            return
        res = self.result
        new = await explore_variants(
            self.llm, self.settings, self.console, res.spec, res.analysis, base.idea, direction,
            target=self.params.per_call * 2, per_call=self.params.per_call, brief=res.brief,
        )
        known = [r.idea for r in res.audited] + [i for i, _ in res.reserve]
        unique = await dedupe_ideas(self.llm, self.settings, new, existing=known)
        if not unique:
            self.console.print("[yellow]Aucune variante réellement nouvelle. Essaie une consigne plus précise.[/yellow]")
            return
        before = len(res.survivors)
        await screen_and_audit(self.llm, self.settings, self.console, res, unique, self.params.pool)
        res.save()
        self.console.print(f"{len(unique)} variantes nouvelles, {len(res.survivors) - before} retenues. Les numéros peuvent avoir changé.")
        self.show_table()

    async def _more(self) -> None:
        res = self.result
        # Lot varié, et loin des idées déjà auditées. Il n'est retiré de la réserve qu'une fois audité: si on est
        # interrompu (Ctrl+C), il y reste.
        batch = await pick_diverse(
            self.llm, self.settings, res.reserve, self.params.pool, existing=[r.idea for r in res.audited]
        )
        if not batch:
            self.console.print("[yellow]Réserve vide. Essaie `variantes N` pour générer de nouvelles pistes.[/yellow]")
            return
        before = len(res.survivors)
        audited = await deep_verify(self.llm, self.settings, self.console, res.spec, batch, res.brief)
        res.audited += audited
        audited_ids = {i.id for i, _ in batch}
        res.reserve = [iv for iv in res.reserve if iv[0].id not in audited_ids]
        res.save()
        self.console.print(f"{len(batch)} idées auditées, {len(res.survivors) - before} retenues.")
        self.show_table()

    def _rejected(self) -> None:
        items = self.result.rejected_items()
        if not items:
            self.console.print("Rien n'a été écarté.")
            return
        table = Table(title="Hors classement", box=box.ROUNDED, border_style="yellow", header_style="bold yellow")
        for col in ("Réf", "Origine", "Idée", "Raison"):
            table.add_column(col, overflow="fold")
        origins = {"audit": "[red]audit[/red]", "filtre": "[dim]filtre[/dim]", "toi": "[cyan]toi[/cyan]"}
        for n, it in enumerate(items, 1):
            reason = it.reason if len(it.reason) <= 220 else it.reason[:217] + "..."
            table.add_row(f"R{n}", origins[it.origin], escape(it.idea.title), escape(reason))
        self.console.print(table)
        self.console.print("[dim]Tu décides : `garder R2` remet une idée dans le classement, même écartée par le programme.[/dim]")

    def _find_rejected(self, ref: str) -> RejectedItem | None:
        items = self.result.rejected_items()
        ref = ref.strip()
        if ref[:1].lower() == "r" and ref[1:].isdigit():
            n = int(ref[1:])
            return items[n - 1] if 1 <= n <= len(items) else None
        return next((it for it in items if it.idea.id.lower() == ref.lower()), None)

    async def _keep(self, ref: str) -> None:
        item = self._find_rejected(ref)
        if item is None:
            self.console.print("[red]Référence inconnue.[/red] Tape `rejetées` pour voir les références (R1, R2...).")
            return
        res, audit = self.result, None
        if item.origin == "filtre":  # jamais auditée: on l'audite maintenant pour lui donner une place dans le classement
            neutral = GateVerdict(violates_red_line=False, illegal_or_harmful=False, plausible=True, score=5,
                                  reason="gardée par toi malgré le filtre")
            audited = await deep_verify(self.llm, self.settings, self.console, res.spec, [(item.idea, neutral)], res.brief)
            if not audited:
                self.console.print("[red]L'audit de cette idée a échoué, réessaie.[/red]")
                return
            audit = audited[0]
        res.keep(item, audit)
        res.save()
        self.console.print(f"[green]Remise dans le classement :[/green] {escape(item.idea.title)}")
        shown = audit or next((r for r in res.audited if r.idea.id == item.idea.id), None)
        if shown and shown.warnings:
            self.console.print(f"[yellow]⚠ Alertes de l'audit : {', '.join(shown.warnings)}[/yellow]")
        self.show_table()

    def _veto(self, arg: str) -> None:
        r = self._pick(arg)
        if not r:
            return
        self.result.veto(r.idea.id)
        self.result.save()
        self.console.print(f"Retirée du classement : {escape(r.idea.title)}  [dim](`rejetées` puis `garder` la remettent)[/dim]")
        self.show_table()

    async def _info(self, text: str) -> None:
        spec = self.result.spec.model_copy(update={"key_facts": [*self.result.spec.key_facts, text]})
        self.console.print("Fait ajouté à la fiche, relance de la résolution...")
        self.result = await solve(self.llm, self.settings, self.console, spec, self.params, self._confirm)
        report = self.result.run_dir / "report.md"
        if report.exists():
            self.console.print(Markdown(report.read_text(encoding="utf-8")))
        self.show_table()

    async def _confirm(self, question: str) -> bool:
        return (await self.read(f"{question} [O/n] ")).strip().lower() not in {"n", "non"}

    async def _web(self, question: str) -> None:
        res = self.result
        if not self.settings.web_enabled:
            self.console.print("[yellow]La recherche web est désactivée (--no-web ou SWARM_WEB=0).[/yellow]")
            return
        answer, res.research = await ask_web(self.llm, self.settings, self.console, res.spec, res.research, question)
        res.save()
        self.console.print(Markdown(answer))
        tail = res.research.sources_markdown(answer) if res.research else ""
        if tail:
            self.console.print(Markdown(tail))

    def _sources(self) -> None:
        sources = self.result.research.sources if self.result.research else []
        if not sources:
            self.console.print("Aucune source web collectée.")
            return
        for s in sources:
            self.console.print(f"{escape(f'[S{s.id}]')} {escape(s.title)} [dim]({s.origin}) {escape(s.url)}[/dim]")

    async def _question(self, text: str) -> None:
        res = self.result
        past = "\n".join(f"USER: {q}\nADVISOR: {a}" for q, a in self.history[-HISTORY_TURNS:]) or "(none)"
        user = (
            f"{res.spec.to_prompt()}\n\n{res.analysis.to_prompt()}{brief_block(res.brief)}\n\nSOLUTIONS:\n{self._context()}\n\n"
            f"CONVERSATION SO FAR:\n{past}\n\nUSER: {text}"
        )
        system = ADVISOR.format(language=self.settings.language)
        with self.console.status("Réflexion..."):
            answer = await self.llm.text(self.settings.strong_model, system, user, tag="conseiller")
        self.history.append((text, answer))
        self.console.print(Markdown(answer))
