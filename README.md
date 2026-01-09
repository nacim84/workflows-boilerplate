# Agentic Workflows Boilerplate

Ce projet est un boilerplate générique conçu pour structurer et déployer des workflows agentiques IA fiables. Il repose sur une architecture à trois couches qui sépare le raisonnement probabiliste des LLM de l'exécution déterministe du code.

**Alternative by-code à N8N** : Ce projet permet de créer des automatisations et workflows avec du code Python, des agents IA avancés, et un écosystème complet d'outils et MCPs pour répondre à des besoins business concrets et rentables.

## 🏗 L'Architecture à 3 Couches

Pour maximiser la fiabilité, ce système sépare les responsabilités :

1.  **Couche 1 : Directive (Le "Quoi")**
    *   Située dans `directives/`.
    *   Procédures Opérationnelles Standard (SOP) en Markdown.
    *   Définit les objectifs, les entrées/sorties et les outils à utiliser.

2.  **Couche 2 : Orchestration (La Décision)**
    *   C'est l'Agent (LLM).
    *   Lit les directives, sélectionne les outils d'exécution, gère les erreurs et met à jour les instructions en fonction des apprentissages.

3.  **Couche 3 : Exécution (Le "Comment")**
    *   Située dans `execution/`.
    *   Scripts Python déterministes.
    *   Gère les appels API, le traitement de données et les interactions système de manière fiable et testable.

## 📂 Structure du Projet

```text
.
├── directives/                  # Instructions et SOPs (Markdown)
│   ├── TECHNICAL_SPECS.md       # Spécifications techniques complètes
│   ├── mcp-servers-guide.md     # Guide d'implémentation MCP
│   └── workflow_*.md            # SOPs de workflows spécifiques
├── execution/                   # Scripts Python (Outils déterministes)
│   ├── core/                    # Configuration et utilitaires
│   ├── agents/                  # Implémentations d'agents
│   ├── workflows/               # Orchestration de workflows
│   ├── tools/                   # Outils réutilisables
│   └── mcp_servers/             # Serveurs MCP personnalisés
├── tests/                       # Tests unitaires et d'intégration
├── .tmp/                        # Fichiers intermédiaires (non commités)
├── .env                         # Variables d'environnement et clés API
├── pyproject.toml               # Configuration UV et dépendances
├── .python-version              # Version Python (3.12+)
├── AGENTS.md                    # Instructions système pour l'Agent
└── README.md                    # Documentation du projet
```

## 🚀 Principes de Fonctionnement

*   **Priorité aux Outils :** Toujours vérifier si un script existe dans `execution/` avant d'en créer un nouveau.
*   **Auto-réparation (Self-healing) :** En cas d'erreur, l'agent analyse la stack trace, corrige le script d'exécution et met à jour la directive correspondante pour éviter la récurrence du problème.
*   **Directives Vivantes :** Les documents dans `directives/` évoluent avec le temps pour inclure les limites d'API découvertes, les cas limites et les meilleures approches.
*   **Fiabilité Déterministe :** En déportant la complexité vers du code (Layer 3), on garantit un taux de réussite bien plus élevé qu'en laissant le LLM manipuler les données directement.

## 🛠 Installation et Usage

### Prérequis

- Python 3.11+ (recommandé : 3.12+)
- UV (gestionnaire de paquets Astral)

### Installation

```bash
# 1. Installer UV (si pas déjà installé)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# ou
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Cloner le projet
git clone https://github.com/yourusername/workflows-boilerplate.git
cd workflows-boilerplate

# 3. Installer les dépendances
uv sync

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

### Utilisation

1. **Créer une directive** : Ajoutez vos SOPs dans `directives/workflow_name.md`
2. **Développer les outils** : Implémentez les scripts Python dans `execution/`
3. **Configurer les agents** : Créez vos agents dans `execution/agents/`
4. **Orchestrer** : L'agent LLM utilisera `AGENTS.md` pour coordonner les workflows

### Ajouter des dépendances

```bash
# Dépendance de production
uv add nom-du-package

# Dépendance de développement
uv add --dev pytest

# Tout est automatiquement ajouté à pyproject.toml et verrouillé dans uv.lock
```

## 📚 Documentation

### Guides Techniques

- **[Spécifications Techniques](directives/TECHNICAL_SPECS.md)** : Stack technique complète, frameworks agentiques (LangGraph, CrewAI, AutoGen, etc.), patterns et best practices
- **[Guide MCP Servers](directives/mcp-servers-guide.md)** : Implémentation de serveurs Model Context Protocol pour étendre les capacités des agents
- **[Instructions Agent](AGENTS.md)** : Directives système pour l'orchestration par les LLMs

### Stack Technique Principal

- **Python 3.12** avec **UV** (Astral) pour gestion de dépendances
- **Frameworks Agentiques** : LangGraph, CrewAI, LlamaIndex, Pydantic AI
- **LLM Providers** : Anthropic Claude, OpenAI, LiteLLM (abstraction unifiée)
- **MCP** : Model Context Protocol pour intégrations externes
- **Infrastructure** : FastAPI, Redis, PostgreSQL, Celery
- **Observabilité** : Structlog, LangFuse, Prometheus

### Frameworks Recommandés par Use Case

| Use Case | Framework |
|----------|-----------|
| Workflows complexes stateful | **LangGraph** |
| Équipes d'agents collaboratifs | **CrewAI** |
| RAG et knowledge bases | **LlamaIndex** |
| Type-safety et validation stricte | **Pydantic AI** |

---

## 🎯 Objectifs du Projet

Ce boilerplate vise à :
- Transformer l'IA d'un simple moteur de chat en un système opérationnel robuste
- Fournir une alternative **by-code** à N8N avec agents IA avancés
- Créer un écosystème complet d'outils et MCPs pour le business
- Permettre de bootstrapper rapidement des projets d'automatisation rentables

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez les spécifications techniques avant de contribuer.

## 📄 Licence

MIT
