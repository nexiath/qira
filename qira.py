#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum-Inspired Refactoring Assistant (QIRA) - Version Avancée
================================================================
Cet outil analyse le code source Python pour en extraire des métriques détaillées
pour les fonctions et les classes, génère des suggestions de refactoring adaptées
(p. ex. extraction, simplification, aplatissage, décomposition, ajout de documentation),
puis utilise un algorithme de recuit simulé (inspiré des principes quantiques) pour
sélectionner l'ensemble optimal des améliorations à apporter.

Usage :
    python qira.py --file chemin/vers/fichier.py [--length_threshold 30]
                   [--complexity_threshold 8] [--depth_threshold 3]
                   [--methods_threshold 5] [--class_complexity_threshold 5.0]
                   [--doc_required] [--iterations 3000]
                   [--debug] [--output rapport.json]

Options :
    --file                     : Chemin vers le fichier source Python à analyser.
    --length_threshold         : Seuil de longueur (en lignes) pour fonctions/classes (défaut : 30).
    --complexity_threshold     : Seuil de complexité cyclomatique pour fonctions (défaut : 8).
    --depth_threshold          : Seuil de profondeur d'imbrication pour fonctions (défaut : 3).
    --methods_threshold        : Seuil du nombre de méthodes pour une classe (défaut : 5).
    --class_complexity_threshold: Seuil de complexité moyenne des méthodes pour une classe (défaut : 5.0).
    --doc_required             : Si présent, vérifie la présence de docstrings pour fonctions/classes.
    --iterations               : Nombre maximum d'itérations pour le recuit simulé (défaut : 3000).
    --debug                    : Active le mode débogage pour afficher des informations détaillées.
    --output                   : Chemin vers un fichier JSON pour exporter le rapport.
    
Auteur : nexiath
Date : 2025-02-17
"""

import ast
import math
import random
import sys
import argparse
import logging
import json
from dataclasses import dataclass
from typing import List

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Définition des structures de données pour stocker les métriques ---

@dataclass
class FunctionMetrics:
    name: str
    start_line: int
    end_line: int
    length: int
    complexity: int
    max_depth: int
    num_params: int
    has_docstring: bool

@dataclass
class ClassMetrics:
    name: str
    start_line: int
    end_line: int
    length: int
    num_methods: int
    average_complexity: float
    has_docstring: bool

@dataclass
class RefactoringCandidate:
    scope: str   # "function" ou "class"
    name: str
    candidate_type: str  # ex. "extraction", "simplification", "flattening", "documentation", "decomposition"
    metric: float
    threshold: float
    cost: float
    improvement: float
    description: str

# --- Fonctions d'analyse du code source ---

def compute_cyclomatic_complexity(node: ast.AST) -> int:
    """
    Calcule la complexité cyclomatique en comptant les nœuds de contrôle.
    Les structures considérées : if, for, while, try, with, boolops, except, comprehension.
    """
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                              ast.BoolOp, ast.ExceptHandler, ast.comprehension)):
            count += 1
    return count

def compute_max_depth(node: ast.AST, current_depth=0) -> int:
    """
    Calcule la profondeur maximale d'imbrication d'un nœud AST.
    """
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        child_depth = compute_max_depth(child, current_depth + 1)
        max_depth = max(max_depth, child_depth)
    return max_depth

def analyze_functions(tree: ast.AST) -> List[FunctionMetrics]:
    """
    Parcourt l'arbre AST pour extraire les métriques de chaque fonction.
    """
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', node.lineno)
            length = end_line - start_line + 1
            complexity = compute_cyclomatic_complexity(node)
            max_depth = compute_max_depth(node)
            num_params = len(node.args.args) if hasattr(node.args, 'args') else 0
            has_docstring = (ast.get_docstring(node) is not None)
            functions.append(FunctionMetrics(
                name=node.name,
                start_line=start_line,
                end_line=end_line,
                length=length,
                complexity=complexity,
                max_depth=max_depth,
                num_params=num_params,
                has_docstring=has_docstring
            ))
    return functions

def analyze_classes(tree: ast.AST) -> List[ClassMetrics]:
    """
    Parcourt l'arbre AST pour extraire les métriques de chaque classe.
    """
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', node.lineno)
            length = end_line - start_line + 1
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            num_methods = len(methods)
            total_complexity = sum(compute_cyclomatic_complexity(m) for m in methods)
            average_complexity = total_complexity / num_methods if num_methods > 0 else 0
            has_docstring = (ast.get_docstring(node) is not None)
            classes.append(ClassMetrics(
                name=node.name,
                start_line=start_line,
                end_line=end_line,
                length=length,
                num_methods=num_methods,
                average_complexity=average_complexity,
                has_docstring=has_docstring
            ))
    return classes

# --- Génération des suggestions de refactoring ---

def generate_function_candidates(functions: List[FunctionMetrics],
                                 length_threshold: int,
                                 complexity_threshold: int,
                                 depth_threshold: int,
                                 doc_threshold: bool) -> List[RefactoringCandidate]:
    """
    Pour chaque fonction, génère des suggestions de refactoring en fonction de :
      - La longueur
      - La complexité cyclomatique
      - La profondeur d'imbrication
      - La présence d'une docstring (si requis)
    """
    candidates = []
    for func in functions:
        # Extraction si fonction trop longue
        if func.length > length_threshold:
            cost = func.length - length_threshold
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La fonction '{func.name}' est trop longue ({func.length} lignes). "
                           "Envisagez d'extraire des sous-fonctions pour améliorer la lisibilité.")
            candidates.append(RefactoringCandidate(
                scope="function",
                name=func.name,
                candidate_type="extraction",
                metric=func.length,
                threshold=length_threshold,
                cost=cost,
                improvement=improvement,
                description=description
            ))
        # Simplification si complexité élevée
        if func.complexity > complexity_threshold:
            cost = (func.complexity - complexity_threshold) * 10
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La fonction '{func.name}' a une complexité cyclomatique élevée ({func.complexity}). "
                           "Envisagez de simplifier la logique ou de refactoriser les conditions.")
            candidates.append(RefactoringCandidate(
                scope="function",
                name=func.name,
                candidate_type="simplification",
                metric=func.complexity,
                threshold=complexity_threshold,
                cost=cost,
                improvement=improvement,
                description=description
            ))
        # Aplatissement si profondeur trop élevée
        if func.max_depth > depth_threshold:
            cost = (func.max_depth - depth_threshold) * 5
            improvement = cost * random.uniform(0.2, 0.35)
            description = (f"La fonction '{func.name}' présente une imbrication trop profonde (profondeur {func.max_depth}). "
                           "Envisagez d'aplatir la structure pour améliorer la clarté.")
            candidates.append(RefactoringCandidate(
                scope="function",
                name=func.name,
                candidate_type="flattening",
                metric=func.max_depth,
                threshold=depth_threshold,
                cost=cost,
                improvement=improvement,
                description=description
            ))
        # Documentation manquante
        if not func.has_docstring and doc_threshold:
            cost = 5  # coût fixe pour absence de docstring
            improvement = 5
            description = (f"La fonction '{func.name}' ne possède pas de docstring. "
                           "L'ajout d'une documentation améliorerait la maintenabilité.")
            candidates.append(RefactoringCandidate(
                scope="function",
                name=func.name,
                candidate_type="documentation",
                metric=0,
                threshold=0,
                cost=cost,
                improvement=improvement,
                description=description
            ))
    return candidates

def generate_class_candidates(classes: List[ClassMetrics],
                              length_threshold: int,
                              methods_threshold: int,
                              complexity_threshold: float,
                              doc_threshold: bool) -> List[RefactoringCandidate]:
    """
    Pour chaque classe, génère des suggestions de refactoring en fonction de :
      - La longueur totale de la classe
      - Le nombre de méthodes
      - La complexité moyenne des méthodes
      - La présence d'une docstring (si requis)
    """
    candidates = []
    for cls in classes:
        # Décomposition si classe trop longue
        if cls.length > length_threshold:
            cost = cls.length - length_threshold
            improvement = cost * random.uniform(0.25, 0.45)
            description = (f"La classe '{cls.name}' est trop longue ({cls.length} lignes). "
                           "Envisagez de la diviser ou d'extraire des sous-classes.")
            candidates.append(RefactoringCandidate(
                scope="class",
                name=cls.name,
                candidate_type="decomposition",
                metric=cls.length,
                threshold=length_threshold,
                cost=cost,
                improvement=improvement,
                description=description
            ))
        # Trop de méthodes
        if cls.num_methods > methods_threshold:
            cost = cls.num_methods - methods_threshold
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La classe '{cls.name}' contient trop de méthodes ({cls.num_methods}). "
                           "Envisagez de refactoriser en plusieurs classes pour améliorer la cohésion.")
            candidates.append(RefactoringCandidate(
                scope="class",
                name=cls.name,
                candidate_type="method_refactoring",
                metric=cls.num_methods,
                threshold=methods_threshold,
                cost=cost,
                improvement=improvement,
                description=description
            ))
        # Complexité moyenne trop élevée
        if cls.average_complexity > complexity_threshold:
            cost = (cls.average_complexity - complexity_threshold) * 10
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La classe '{cls.name}' a une complexité moyenne des méthodes élevée "
                           f"({cls.average_complexity:.2f}). Envisagez d'optimiser certaines méthodes.")
            candidates.append(RefactoringCandidate(
                scope="class",
                name=cls.name,
                candidate_type="complexity_refactoring",
                metric=cls.average_complexity,
                threshold=complexity_threshold,
                cost=cost,
                improvement=improvement,
                description=description
            ))
        # Documentation manquante
        if not cls.has_docstring and doc_threshold:
            cost = 5
            improvement = 5
            description = (f"La classe '{cls.name}' ne possède pas de docstring. "
                           "L'ajout d'une documentation améliorerait la maintenabilité.")
            candidates.append(RefactoringCandidate(
                scope="class",
                name=cls.name,
                candidate_type="documentation",
                metric=0,
                threshold=0,
                cost=cost,
                improvement=improvement,
                description=description
            ))
    return candidates

# --- Optimisation par recuit simulé ---

def cost_function(state: List[int], candidates: List[RefactoringCandidate]) -> float:
    """
    Calcule le coût total d'un état.
    Pour chaque candidate, si le refactoring est appliqué (bit = 1), le coût est réduit
    par l'amélioration estimée (mais ne descend pas en dessous de zéro).
    """
    total = 0
    for applied, cand in zip(state, candidates):
        if applied:
            total += max(cand.cost - cand.improvement, 0)
        else:
            total += cand.cost
    return total

def perturb_state(state: List[int]) -> List[int]:
    """
    Applique une perturbation aléatoire sur l'état en inversant le bit d'une candidate choisi au hasard.
    """
    new_state = state.copy()
    idx = random.randint(0, len(state)-1)
    new_state[idx] = 1 - new_state[idx]
    return new_state

def simulated_annealing(candidates: List[RefactoringCandidate],
                        initial_state: List[int],
                        initial_temp: float = 100.0,
                        cooling_rate: float = 0.97,
                        iterations: int = 3000,
                        debug: bool = False) -> (List[int], float):
    """
    Algorithme de recuit simulé pour optimiser la sélection des suggestions de refactoring.
    """
    current_state = initial_state.copy()
    current_cost = cost_function(current_state, candidates)
    best_state = current_state.copy()
    best_cost = current_cost
    temp = initial_temp

    for i in range(iterations):
        new_state = perturb_state(current_state)
        new_cost = cost_function(new_state, candidates)
        delta = new_cost - current_cost
        if delta < 0 or math.exp(-delta / temp) > random.random():
            current_state = new_state
            current_cost = new_cost
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
        temp *= cooling_rate
        if debug and i % 100 == 0:
            logger.debug(f"Iter {i:4d}, Temp: {temp:7.4f}, Current Cost: {current_cost:7.2f}, Best Cost: {best_cost:7.2f}")
        if temp < 1e-3:
            break
    return best_state, best_cost

# --- Génération du rapport ---

def generate_report(candidates: List[RefactoringCandidate], state: List[int]) -> List[dict]:
    """
    Génère un rapport détaillé sous forme de liste de dictionnaires pour les suggestions
    de refactoring retenues après optimisation.
    """
    report = []
    for applied, cand in zip(state, candidates):
        if applied:
            report.append({
                "scope": cand.scope,
                "name": cand.name,
                "candidate_type": cand.candidate_type,
                "metric": cand.metric,
                "threshold": cand.threshold,
                "cost": cand.cost,
                "improvement": cand.improvement,
                "description": cand.description
            })
    # Tri par scope puis par type de refactoring
    report.sort(key=lambda x: (x["scope"], x["candidate_type"]))
    return report

def write_report_json(report: List[dict], output_file: str):
    """
    Écrit le rapport dans un fichier JSON.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        logger.info(f"Rapport écrit dans le fichier {output_file}")
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture du rapport JSON: {e}")

# --- Parsing des arguments et fonction principale ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantum-Inspired Refactoring Assistant (QIRA) - Version Avancée")
    parser.add_argument("--file", required=True, help="Chemin vers le fichier source Python à analyser.")
    parser.add_argument("--length_threshold", type=int, default=30, help="Seuil de longueur pour fonctions/classes (lignes).")
    parser.add_argument("--complexity_threshold", type=int, default=8, help="Seuil de complexité cyclomatique pour fonctions.")
    parser.add_argument("--depth_threshold", type=int, default=3, help="Seuil de profondeur d'imbrication pour fonctions.")
    parser.add_argument("--methods_threshold", type=int, default=5, help="Seuil du nombre de méthodes pour une classe.")
    parser.add_argument("--class_complexity_threshold", type=float, default=5.0, help="Seuil de complexité moyenne des méthodes pour une classe.")
    parser.add_argument("--doc_required", action="store_true", help="Vérifier la présence de docstrings pour fonctions/classes.")
    parser.add_argument("--iterations", type=int, default=3000, help="Nombre maximum d'itérations pour le recuit simulé.")
    parser.add_argument("--debug", action="store_true", help="Active le mode débogage.")
    parser.add_argument("--output", help="Fichier de sortie pour le rapport JSON.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier: {e}")
        sys.exit(1)

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.error(f"Erreur de syntaxe dans le code source: {e}")
        sys.exit(1)

    logger.info("Analyse de l'arbre AST...")
    functions = analyze_functions(tree)
    classes = analyze_classes(tree)
    logger.info(f"{len(functions)} fonction(s) détectée(s).")
    logger.info(f"{len(classes)} classe(s) détectée(s).")

    logger.info("Génération des suggestions de refactoring pour les fonctions...")
    func_candidates = generate_function_candidates(
        functions,
        length_threshold=args.length_threshold,
        complexity_threshold=args.complexity_threshold,
        depth_threshold=args.depth_threshold,
        doc_threshold=args.doc_required
    )
    logger.info(f"{len(func_candidates)} suggestion(s) générée(s) pour les fonctions.")

    logger.info("Génération des suggestions de refactoring pour les classes...")
    class_candidates = generate_class_candidates(
        classes,
        length_threshold=args.length_threshold,
        methods_threshold=args.methods_threshold,
        complexity_threshold=args.class_complexity_threshold,
        doc_threshold=args.doc_required
    )
    logger.info(f"{len(class_candidates)} suggestion(s) générée(s) pour les classes.")

    all_candidates = func_candidates + class_candidates
    if not all_candidates:
        logger.info("Aucune suggestion de refactoring nécessaire détectée.")
        sys.exit(0)

    if args.debug:
        for cand in all_candidates:
            logger.debug(cand)

    initial_state = [0] * len(all_candidates)
    logger.info("Optimisation par recuit simulé en cours...")
    best_state, best_cost = simulated_annealing(all_candidates, initial_state,
                                                  iterations=args.iterations,
                                                  debug=args.debug)
    logger.info(f"Optimisation terminée. Coût optimal: {best_cost:.2f}")

    report = generate_report(all_candidates, best_state)
    if report:
        logger.info("Rapport final des suggestions de refactoring:")
        for item in report:
            logger.info(f"{item['scope'].capitalize()} '{item['name']}' - {item['candidate_type'].capitalize()}: {item['description']} "
                        f"(Coût: {item['cost']:.2f}, Amélioration: {item['improvement']:.2f})")
    else:
        logger.info("Aucune suggestion retenue après optimisation.")

    if args.output:
        write_report_json(report, args.output)

if __name__ == "__main__":
    main()
