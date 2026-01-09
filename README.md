# Agentic Workflows Boilerplate

Ce projet est un boilerplate générique conçu pour structurer et déployer des workflows agentiques IA fiables. Il repose sur une architecture à trois couches qui sépare le raisonnement probabiliste des LLM de l'exécution déterministe du code.

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
├── directives/      # Instructions et SOPs (Markdown)
├── execution/       # Scripts Python (Outils déterministes)
├── .tmp/            # Fichiers intermédiaires (non commités)
├── .env             # Variables d'environnement et clés API
├── AGENTS.md        # Instructions système pour l'Agent
└── README.md        # Documentation du projet
```

## 🚀 Principes de Fonctionnement

*   **Priorité aux Outils :** Toujours vérifier si un script existe dans `execution/` avant d'en créer un nouveau.
*   **Auto-réparation (Self-healing) :** En cas d'erreur, l'agent analyse la stack trace, corrige le script d'exécution et met à jour la directive correspondante pour éviter la récurrence du problème.
*   **Directives Vivantes :** Les documents dans `directives/` évoluent avec le temps pour inclure les limites d'API découvertes, les cas limites et les meilleures approches.
*   **Fiabilité Déterministe :** En déportant la complexité vers du code (Layer 3), on garantit un taux de réussite bien plus élevé qu'en laissant le LLM manipuler les données directement.

## 🛠 Installation et Usage

1.  **Configuration :** Créez un fichier `.env` à la racine pour vos clés API.
2.  **Directives :** Ajoutez vos SOPs dans le dossier `directives/`.
3.  **Exécution :** Développez vos scripts de traitement dans `execution/`.
4.  **Interactions :** L'agent utilisera `AGENTS.md` comme contexte de base pour orchestrer vos workflows.

---
*Ce boilerplate vise à transformer l'IA d'un simple moteur de chat en un système opérationnel robuste.*
