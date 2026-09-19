import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LOG = "swarm_solver"
TRACE = "swarm_solver.trace"  # prompts et réponses complets


def setup_logging(console: Console, verbosity: int, log_file: Path) -> None:
    """0 = silencieux. 1 (-v) = appels, durées, verdicts. 2 (-vv) = aussi prompts et réponses complets.

    Dès -v, tout (prompts et réponses compris) est aussi écrit dans `log_file` pour les analyses a posteriori.
    """
    logger = logging.getLogger(LOG)
    logger.handlers.clear()
    logger.propagate = False
    if verbosity < 1:
        logger.setLevel(logging.WARNING)
        logger.addHandler(RichHandler(console=console, show_path=False, markup=False))
        return

    logger.setLevel(logging.DEBUG)
    screen = RichHandler(console=console, show_path=False, markup=False, log_time_format="%H:%M:%S")
    if verbosity < 2:
        screen.addFilter(lambda record: record.name != TRACE)
    logger.addHandler(screen)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    to_file = logging.FileHandler(log_file, encoding="utf-8")
    to_file.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(to_file)
