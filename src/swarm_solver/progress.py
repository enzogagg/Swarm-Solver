import asyncio
import logging
from collections.abc import Awaitable
from typing import TypeVar

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .logs import LOG

T = TypeVar("T")

log = logging.getLogger(f"{LOG}.progress")


async def gather_tolerant(
    console: Console, description: str, coros: list[Awaitable[T]]
) -> tuple[list[T], int, str | None]:
    """Exécute tous les appels en parallèle. Un échec isolé ne doit pas tuer un run de plusieurs minutes.

    Retourne (résultats réussis, nombre d'échecs, dernière erreur).
    """
    failures = 0
    last_error: str | None = None
    columns = (SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
    with Progress(*columns, console=console) as progress:
        task = progress.add_task(description, total=len(coros))

        async def guarded(coro: Awaitable[T]) -> T | None:
            nonlocal failures, last_error
            try:
                return await coro
            except Exception as e:  # noqa: BLE001 - on veut tout tolérer ici
                failures += 1
                last_error = f"{type(e).__name__}: {e}"
                log.debug("%s: échec: %s", description, last_error)
                return None
            finally:
                progress.advance(task)

        out = await asyncio.gather(*(guarded(c) for c in coros))
    return [r for r in out if r is not None], failures, last_error
