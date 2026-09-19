# Swarm Solver

Résolveur de problèmes complexes multi-agents, **100 % local via Ollama**.

Tu décris un problème, l'outil t'interroge, cherche sur le web, génère des centaines d'idées en parallèle avec de
petits agents, les vérifie avec d'autres, et te livre un classement argumenté sur lequel tu peux rebondir. Il sert
pour tout problème où tu as besoin d'options plutôt que d'une seule réponse :

- technique : « diviser par deux le temps de build de la CI d'un monorepo », « migrer une base sans interruption »
- business : « trouver 50 premiers clients sans budget marketing », « négocier un contrat face à un fournisseur dominant »
- organisation et carrière : « débloquer un projet bloqué par un décideur », « obtenir une augmentation après un refus »
- personnel : « réduire une dépense de moitié en six mois », « choisir entre deux offres »

```
interview -> recherche web -> analyse -> essaim -> dédup -> filtre rapide -> 3 audits -> rapport -> session
  (14B)        (8B + 14B)       (14B)     (8B, //)  (embed)      (8B)          (14B)       (14B)     interactive
```

- **Recherche web** : voir la section dédiée plus bas. Produit un dossier de faits sourcés `[S1]`, `[S2]`...
  qui alimente l'analyse, l'essaim, les audits (légalité et faisabilité) et le rapport.
- **Analyse** : avant de chercher des solutions, un stratège cartographie les acteurs, leurs intérêts, pourquoi la
  situation tient et où sont les leviers. Tout l'essaim reçoit cette analyse.
- **Essaim** : 15 angles généraux en parallèle (plus petite chose qui marche, contraintes à lever, expériences
  réversibles, inversion, analogies, couverture d'hypothèse...). Quand l'analyse détecte que quelqu'un s'oppose à
  l'objectif, l'entrave ou profite du problème, 6 angles « rapport de force » s'y ajoutent (incitations, hardball
  licite, visibilité des faits, escalade...). Les idées peuvent être dures, cyniques ou politiquement incorrectes :
  le ton n'est pas un critère.
- **Entonnoir** : ~1000 idées brutes -> dédup -> filtre rapide (1 appel/idée) -> audit approfondi du meilleur lot.
  Un gros modèle sur 1000 idées serait trop lent en local.

## Mise en route (exemple : GPU 16 Go, type RTX 5080)

```bash
ollama pull qwen3:8b && ollama pull qwen3:14b && ollama pull nomic-embed-text
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 ollama serve   # dans un autre terminal

uv sync --extra dev
uv run swarm-solver doctor                       # vérifie Ollama + modèles
uv run swarm-solver solve                        # interview, résolution, puis session interactive
uv run swarm-solver solve "ma description" --ideas 2000
uv run swarm-solver solve --spec runs/<date>/problem.json   # relance sans interview
uv run swarm-solver solve -v                     # mode verbose (voir ci-dessous)
```

## Recherche web

```
fiche -> requêtes anonymisées (14B) -> recherche -> lecture des pages -> faits sourcés (8B) -> dossier cité (14B)
```

| Source | Comment | Limites |
|---|---|---|
| Web | DuckDuckGo via `ddgs` (aucune clé) ou une instance SearXNG auto-hébergée (`SWARM_SEARCH_PROVIDER=searxng`, `SWARM_SEARXNG_URL=http://localhost:8080`, format JSON activé dans son `settings.yml`) | résultats limités par le moteur |
| Reddit | recherche `site:reddit.com`, on ne lit que l'**extrait** | Reddit répond 403 aux accès anonymes : fils et commentaires non lisibles sans l'API officielle |
| Hacker News | API publique Algolia, texte complet | surtout tech, start-up, management |

Pas d'Instagram, de Facebook ni de dark web : leurs CGU interdisent le scraping, il est très peu fiable, ou l'apport
pour un problème de stratégie est nul.

**Ce qui part sur internet** : uniquement les requêtes, qui doivent décrire le *type* de situation, jamais des noms,
ton employeur ou ta ville. Elles sont affichées avant l'envoi. Ce garde-fou repose sur le modèle : relis-les, et
utilise `--no-web` si le sujet est trop identifiable. Les pages, elles, sont analysées localement.

**Garde-fous de la recherche**
- Chaque fait doit porter une **citation mot pour mot** de la page, vérifiée par le code : sinon il est rejeté.
- L'extracteur ne voit **jamais** ta fiche (un petit modèle en recopie les faits comme s'ils venaient de la page).
- Les citations `[S#]` du dossier sont vérifiées : un identifiant inventé supprime le point.
- Le contenu des pages est traité comme donnée non fiable, jamais comme instruction ; aucun modèle n'a d'outil.
- Téléchargements : adresses publiques uniquement (redirections comprises), `robots.txt` respecté, taille bornée.

Le dossier est écrit dans `research.json`, et le rapport se termine par la liste des sources citées.

## Rebondir sur les solutions (session interactive)

Après le rapport, un prompt `swarm>` s'ouvre :

| Commande | Effet |
|---|---|
| `2` | plan détaillé de la solution 2 : jour par jour, messages prêts à copier, signaux, plan B |
| `variantes 2 plus discret` | génère et audite des variantes de la solution 2 (consigne optionnelle) |
| `encore` | audite le lot suivant d'idées gardées en réserve |
| `liste` | réaffiche le classement |
| `rejetées` | ce qui a été écarté (filtre ou audit) et la raison |
| `info le budget est gelé jusqu'en mars` | ajoute un fait à la fiche et relance toute la résolution |
| `recherche limites de débit de l'API X ?` | cherche sur le web pour une question précise, répond avec sources, enrichit le dossier |
| `sources` | liste les sources web collectées |
| texte libre | discussion avec le conseiller, ex. « et si le budget est divisé par deux ? » |
| `q` | quitter |

`--no-followup` désactive la session. `--no-web` (ou `SWARM_WEB=0`) désactive toute recherche sur internet.

## Reprendre un run

Chaque étape est écrite sur disque dès qu'elle est finie (la recherche web l'est même avant sa synthèse). Si un run
s'arrête (Ctrl+C, terminal fermé, panne), `resume` recharge ce qui existe, **ne refait que ce qui manque**, puis
rouvre la session `swarm>` :

```bash
uv run swarm-solver resume                          # le run le plus récent
uv run swarm-solver resume runs/20260919-165549     # un run précis (chemin ou nom)
uv run swarm-solver resume runs/20260919-165549 --ideas 50 --pool 10   # mêmes réglages que le run d'origine
```

- Un run **terminé** est rechargé sans aucun appel au modèle : tu retrouves le rapport, le classement et les commandes
  (`variantes`, `encore`, `recherche`...). Les idées gardées en réserve (`reserve.jsonl`) reviennent avec leur note.
- Un run **interrompu** repart de la première étape absente : par exemple, recherche déjà faite, on enchaîne sur la
  synthèse puis l'analyse, l'essaim, l'audit et le rapport.
- Les fichiers d'une ancienne version sont relus quand c'est possible (une analyse au format périmé est refaite ; la
  note du filtre des idées en réserve, non sauvegardée à l'époque, est remplacée par une note neutre de 5/10).
- L'historique de discussion de la session (`swarm>`) n'est pas conservé : seul l'état du run l'est.

## Interrompre (Ctrl+C)

| Où tu es | Ctrl+C fait |
|---|---|
| **Recherche web** | interrompt la recherche seule et te demande « Continuer la résolution sans le web ? » : `O` poursuit avec les autres étapes, `n` quitte |
| **Une commande de la session** (`variantes`, `encore`, `recherche`, détail...) | annule la commande et te ramène au prompt `swarm>`, sans rien perdre : un lot d'idées interrompu reste en réserve |
| **Toute autre étape** (analyse, essaim, audits, rapport) | arrête le programme proprement (code 130, sans trace d'erreur) ; ce qui a déjà été produit reste dans `runs/<date>/` |

Les requêtes en cours vers Ollama sont annulées avec l'étape. Ce qui avait déjà été collecté par une recherche web
interrompue n'est pas réutilisé.

## Interface

Bandeau de départ (modèles, web, droit), fiche problème en panneau, étapes numérotées (« Étape 3/7 ») avec leur
durée, tableau des solutions avec barres de score colorées (vert, jaune, rouge) et niveau de risque, rapport en
panneau, puis entonnoir en barres proportionnelles et temps passé par étape.

## Mode verbose

| Option | Affiche |
|---|---|
| `-v` | chaque appel LLM (étape, modèle, attente, génération, tokens), chaque verdict du filtre et des audits, nombre de doublons retirés |
| `-vv` | en plus, prompts et réponses complets à l'écran |

Dès `-v`, tout (prompts et réponses complets compris) est aussi écrit dans `runs/debug.log`.

## Sorties

Chaque run écrit dans `runs/<date>/` : `problem.json`, `analysis.json`, `ideas_raw.jsonl`, `ideas_unique.jsonl`,
`research.json`, `audits.jsonl`, `dropped.jsonl` (idées écartées et raison), `reserve.jsonl` (idées en attente
d'audit), `stats.json`, `report.md`, `detail-<id>.md`.

## Réglages (variables d'environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `SWARM_FAST_MODEL` | `qwen3:8b` | essaim + filtre |
| `SWARM_STRONG_MODEL` | `qwen3:14b` | interview, analyse, audits, rapport, conseiller |
| `SWARM_CONCURRENCY` | `4` | requêtes simultanées (aligner sur `OLLAMA_NUM_PARALLEL`) |
| `SWARM_DEDUP_THRESHOLD` | `0.88` | similarité cosinus au-delà de laquelle deux idées sont fusionnées |
| `SWARM_LANGUAGE` | `français` | langue des sorties |
| `SWARM_JURISDICTION` | `France` | droit de référence pour juger la légalité |
| `SWARM_WEB` | `1` | `0` désactive la recherche web |
| `SWARM_SEARCH_PROVIDER` | `ddgs` | `ddgs` ou `searxng` |
| `SWARM_RESEARCH_QUERIES` | `10` | nombre de requêtes du plan |
| `SWARM_MAX_PAGES` | `24` | pages lues au maximum |

## Ce qui est permis, ce qui est écarté

L'objectif de l'utilisateur est conservé tel quel, même inconfortable (remplacer un fournisseur, faire échouer un
concurrent sur un appel d'offres, voir une personne quitter un poste : ce sont des objectifs valides). Ce qui est
borné, ce sont les **moyens**, et seulement sur le critère de la **loi** : l'éthique, le ton, le cynisme, la
manipulation au sens de la persuasion ou du rapport de force, le caractère politiquement incorrect d'une idée ne
comptent pas. Pression, hardball, silence stratégique, exploitation des intérêts de chacun, faits vrais utilisés
contre quelqu'un : tout ce qui est légal est permis.

Sont écartés, par le filtre puis par un auditeur dédié : actes illégaux, harcèlement, menaces, diffamation (fausses
affirmations qui nuisent), preuves fabriquées ou volées, espionnage de comptes ou communications privés,
provocation ou mise en scène d'une faute, violence physique, campagne soutenue pour tourmenter quelqu'un. Ces
exclusions sont codées en dur (`BASELINE_RED_LINES`) et ajoutées à chaque prompt, quoi que produise l'interview. Le **risque seul** n'écarte jamais une idée : il baisse son score et la
classe « audacieuse » dans le rapport. Tout ce qui est écarté reste consultable avec `rejetées`.

## Tests

```bash
uv run pytest
```

Les tests utilisent un faux LLM (aucun modèle requis).

## Pistes suivantes

- Lecture complète des fils Reddit via l'API officielle (clé gratuite à créer).
- Recherche de validation : après le classement, chercher des preuves que les meilleures solutions marchent en pratique.
- Tournoi par paires sur les finalistes plutôt qu'une simple moyenne pondérée.
- Reprise d'un run interrompu à partir de `ideas_unique.jsonl`.
