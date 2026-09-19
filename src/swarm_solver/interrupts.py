"""Ctrl+C propre : annuler une étape sans tuer le programme, et lire le clavier sans bloquer la sortie."""

import asyncio
import signal
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class Interrupted(Exception):
    """L'étape en cours a été annulée par l'utilisateur (Ctrl+C)."""


class UserQuit(Exception):
    """L'utilisateur a choisi d'arrêter le programme."""


async def interruptible(coro: Coroutine[Any, Any, T]) -> T:
    """Exécute `coro` ; Ctrl+C l'annule et lève `Interrupted` au lieu d'arrêter tout le programme.

    Les requêtes en cours (Ollama, web) sont annulées avec elle. Ailleurs, Ctrl+C garde son effet normal: arrêt.
    Hors du thread principal (où Python interdit de poser un gestionnaire de signal), la coroutine tourne telle quelle.
    """
    loop = asyncio.get_running_loop()
    task = loop.create_task(coro)
    if threading.current_thread() is not threading.main_thread():
        return await task

    previous = signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(task.cancel))
    try:
        return await task
    except asyncio.CancelledError:
        # Si c'est NOUS qui étions annulés (arrêt du programme), on propage. Sinon c'est Ctrl+C sur l'étape.
        if asyncio.current_task().cancelling():
            raise
        raise Interrupted from None
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


async def read_line(prompt: str) -> str:
    """`input()` sans bloquer la boucle, dans un thread *daemon*.

    Avec `asyncio.to_thread`, un `input()` en attente empêche le programme de se fermer après Ctrl+C (il faut
    appuyer sur Entrée). Un thread daemon, lui, est abandonné à la sortie.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(setter: Any, value: Any) -> None:
        if not future.done():
            setter(value)

    def worker() -> None:
        try:
            outcome: tuple[Any, Any] = (future.set_result, input(prompt))
        except BaseException as e:  # noqa: BLE001 - EOFError notamment: à transmettre à l'appelant
            outcome = (future.set_exception, e)
        try:
            loop.call_soon_threadsafe(deliver, *outcome)
        except RuntimeError:  # boucle déjà fermée (Ctrl+C): plus personne n'attend la réponse
            pass

    threading.Thread(target=worker, daemon=True).start()
    return await future
