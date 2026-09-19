"""Recherche web : requêtes anonymisées -> pages -> faits sourcés -> dossier cité [S#]."""

from .agent import ask_web, run_research
from .models import Research, ResearchBrief, brief_block

__all__ = ["Research", "ResearchBrief", "ask_web", "brief_block", "run_research"]
