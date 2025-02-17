# QIRA - Quantum-Inspired Refactoring Assistant

QIRA est un outil innovant d'analyse et de refactoring de code inspiré des principes quantiques. Il aide à améliorer la lisibilité, la maintenabilité et la performance du code en proposant automatiquement des suggestions de refactoring basées sur une analyse statique avancée et une optimisation par recuit simulé.

---

## Table des Matières

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exemple](#exemple)
- [Extensibilité](#extensibilité)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Contact](#contact)

---

## Fonctionnalités

- **Analyse statique du code :**  
  Utilise le module `ast` de Python pour extraire des métriques détaillées sur le code source, telles que :
  - Longueur des fonctions et classes
  - Complexité cyclomatique
  - Profondeur d’imbrication
  - Nombre de paramètres
  - Présence de docstrings

- **Génération de suggestions de refactoring :**  
  Propose automatiquement des améliorations telles que :
  - Extraction de sous-fonctions pour les fonctions trop longues
  - Simplification de la logique pour les fonctions à complexité élevée
  - Aplatissement de structures imbriquées trop profondes
  - Ajout de documentation manquante

- **Optimisation par recuit simulé :**  
  Un algorithme inspiré des principes quantiques explore différentes configurations de refactoring pour sélectionner la combinaison optimale, minimisant ainsi le coût global de la dette technique.

- **Export du rapport :**  
  Possibilité d’exporter un rapport détaillé des suggestions de refactoring au format JSON.

- **Modularité et extensibilité :**  
  Bien que QIRA soit initialement conçu pour Python, son architecture modulaire permet de l’adapter à d’autres langages dans le futur.

---

## Installation

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/votreusername/qira.git
   cd qira
   ```

2. **(Optionnel) Créer un environnement virtuel :**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installer les dépendances :**

   QIRA ne nécessite actuellement pas de dépendances externes majeures. Si des dépendances supplémentaires sont ajoutées ultérieurement, installez-les via :

   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation

Exécutez QIRA en ligne de commande en fournissant le chemin vers le fichier Python à analyser :

```bash
python qira.py --file chemin/vers/fichier.py [options]
```

### Options Disponibles

| Option                        | Description                                            | Défaut      |
|-------------------------------|--------------------------------------------------------|-------------|
| `--file`                      | Obligatoire. Chemin vers le fichier source Python.     |             |
| `--length_threshold`          | Seuil de longueur pour fonctions/classes (lignes).     | 30          |
| `--complexity_threshold`      | Seuil de complexité cyclomatique pour fonctions.       | 8           |
| `--depth_threshold`           | Seuil de profondeur d’imbrication pour fonctions.      | 3           |
| `--methods_threshold`         | Seuil du nombre de méthodes pour une classe.           | 5           |
| `--class_complexity_threshold`| Seuil de complexité moyenne des méthodes par classe.   | 5.0         |
| `--doc_required`              | Vérifie la présence de docstrings.                     |             |
| `--iterations`                | Nombre d’itérations pour le recuit simulé.             | 3000        |
| `--debug`                     | Active le mode débogage.                               |             |
| `--output`                    | Chemin pour exporter le rapport JSON.                  |             |

---

## Exemple

Pour analyser un fichier `exemple.py` avec des seuils personnalisés et exporter le rapport en JSON :

```bash
python qira.py --file exemple.py --length_threshold 40 --complexity_threshold 10 --doc_required --output rapport.json
```

---

## Extensibilité

Bien que QIRA soit actuellement conçu pour Python, son architecture est modulaire et peut être étendue à d'autres langages comme :

- **Java** (via Eclipse JDT ou Spoon)
- **JavaScript/TypeScript** (via Babel ou Esprima)
- **C#** (via Roslyn)
- **C/C++** (via Clang et libTooling)

---

## Contribuer

Les contributions sont les bienvenues !  
Si vous souhaitez améliorer QIRA ou ajouter de nouvelles fonctionnalités, n'hésitez pas à :

1. Forker le dépôt.
2. Créer votre branche de fonctionnalité :

   ```bash
   git checkout -b feature/ma-nouvelle-fonctionnalité
   ```

3. Commiter vos modifications :

   ```bash
   git commit -am 'Ajout de ma nouvelle fonctionnalité'
   ```

4. Pousser sur la branche :

   ```bash
   git push origin feature/ma-nouvelle-fonctionnalité
   ```

5. Créer une **Pull Request**.

---

## Licence

Ce projet est sous licence MIT.

---

## Contact

Pour toute question ou suggestion, vous pouvez :

- Ouvrir une [issue GitHub](https://github.com/nexiath/qira/issues)
- Me contacter via [votre.email@example.com](mailto:robin.cassard39@gmail.com)

---

**QIRA - Modernisez et optimisez votre code grâce à l'inspiration quantique !**
