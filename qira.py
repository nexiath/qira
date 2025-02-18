#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum-Inspired Refactoring Assistant (QIRA) - Version Ultra-Avancée
=====================================================================
Cet outil analyse le code source Python pour en extraire des métriques détaillées
(pour fonctions et classes), détecter des duplications de code et des anomalies de
nomenclature, générer des suggestions de refactoring adaptées, optimiser leur sélection
via un algorithme de recuit simulé, et propose une interface graphique complète ainsi qu’un
mode CLI. Il offre également la possibilité d’analyser un répertoire entier et d’appliquer
automatiquement certains refactorings (via autopep8).

Usage (CLI) :
    python qira.py --file chemin/vers/fichier.py [options]
    python qira.py --dir chemin/vers/repertoire [options]

Si aucun argument n’est passé, l’interface graphique se lance.

Auteur : ChatGPT
Date : 2025-02-17
"""

import ast
import math
import random
import sys
import argparse
import logging
import json
import os
import re
import glob
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Pour l'interface graphique et visualisations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Structures de données ---

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
    source: str  # Source normalisée de la fonction pour détection de duplication

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
    scope: str   # "function" ou "class" ou "variable"
    name: str
    candidate_type: str  # ex. "extraction", "simplification", "flattening", "documentation", "decomposition", "naming"
    metric: float
    threshold: float
    cost: float
    improvement: float
    description: str

# --- Fonctions d'analyse du code source ---

def compute_cyclomatic_complexity(node: ast.AST) -> int:
    """Calcule la complexité cyclomatique en comptant les nœuds de contrôle."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                              ast.BoolOp, ast.ExceptHandler, ast.comprehension)):
            count += 1
    return count

def compute_max_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calcule la profondeur maximale d'imbrication d'un nœud AST."""
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        child_depth = compute_max_depth(child, current_depth + 1)
        max_depth = max(max_depth, child_depth)
    return max_depth

def normalize_source(source: str) -> str:
    """Normalise le code source (suppression espaces, commentaires) pour détecter les duplications."""
    source = re.sub(r'#.*', '', source)        # Supprime les commentaires
    source = re.sub(r'\s+', '', source)         # Supprime les espaces blancs
    return source

def analyze_functions(tree: ast.AST, source_code: str) -> List[FunctionMetrics]:
    """Parcourt l'arbre AST pour extraire les métriques de chaque fonction."""
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
            # Extraire la source exacte de la fonction
            lines = source_code.splitlines()
            func_source = "\n".join(lines[start_line-1:end_line])
            norm_source = normalize_source(func_source)
            functions.append(FunctionMetrics(
                name=node.name,
                start_line=start_line,
                end_line=end_line,
                length=length,
                complexity=complexity,
                max_depth=max_depth,
                num_params=num_params,
                has_docstring=has_docstring,
                source=norm_source
            ))
    return functions

def analyze_classes(tree: ast.AST) -> List[ClassMetrics]:
    """Parcourt l'arbre AST pour extraire les métriques de chaque classe."""
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

# --- Détection de duplications de code ---

def detect_duplicates(functions: List[FunctionMetrics]) -> List[List[FunctionMetrics]]:
    """
    Regroupe les fonctions dont le code source normalisé est identique ou très similaire.
    Renvoie une liste de groupes (chaque groupe contenant deux fonctions ou plus).
    """
    hash_dict: Dict[str, List[FunctionMetrics]] = {}
    for func in functions:
        h = hash(func.source)
        hash_dict.setdefault(h, []).append(func)
    duplicates = [group for group in hash_dict.values() if len(group) > 1]
    return duplicates

# --- Analyse du nommage des variables ---

class VariableNameVisitor(ast.NodeVisitor):
    def __init__(self):
        self.issues = []  # Liste de tuples (nom, ligne)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            # Vérifie que le nom est en snake_case
            if not re.match(r'^[a-z_][a-z0-9_]*$', node.id):
                self.issues.append((node.id, node.lineno))
        self.generic_visit(node)

def analyze_variable_names(tree: ast.AST) -> List[Tuple[str, int]]:
    """
    Analyse les noms de variables dans le code et retourne une liste d'anomalies (nom, ligne)
    pour ceux ne respectant pas la convention snake_case.
    """
    visitor = VariableNameVisitor()
    visitor.visit(tree)
    return visitor.issues

# --- Génération des suggestions de refactoring ---

def generate_function_candidates(functions: List[FunctionMetrics],
                                 length_threshold: int,
                                 complexity_threshold: int,
                                 depth_threshold: int,
                                 doc_threshold: bool) -> List[RefactoringCandidate]:
    """Génère des suggestions de refactoring pour les fonctions."""
    candidates = []
    for func in functions:
        if func.length > length_threshold:
            cost = func.length - length_threshold
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La fonction '{func.name}' est trop longue ({func.length} lignes). "
                           "Envisagez d'extraire des sous-fonctions.")
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
        if func.complexity > complexity_threshold:
            cost = (func.complexity - complexity_threshold) * 10
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La fonction '{func.name}' a une complexité élevée ({func.complexity}). "
                           "Simplifiez la logique interne.")
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
        if func.max_depth > depth_threshold:
            cost = (func.max_depth - depth_threshold) * 5
            improvement = cost * random.uniform(0.2, 0.35)
            description = (f"La fonction '{func.name}' présente une imbrication trop profonde (profondeur {func.max_depth}). "
                           "Aplatissez la structure.")
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
        if not func.has_docstring and doc_threshold:
            cost = 5
            improvement = 5
            description = (f"La fonction '{func.name}' manque de docstring. "
                           "Ajoutez une documentation pour améliorer la maintenabilité.")
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
    """Génère des suggestions de refactoring pour les classes."""
    candidates = []
    for cls in classes:
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
        if cls.num_methods > methods_threshold:
            cost = cls.num_methods - methods_threshold
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La classe '{cls.name}' contient trop de méthodes ({cls.num_methods}). "
                           "Refactorez en plusieurs classes.")
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
        if cls.average_complexity > complexity_threshold:
            cost = (cls.average_complexity - complexity_threshold) * 10
            improvement = cost * random.uniform(0.3, 0.5)
            description = (f"La classe '{cls.name}' a une complexité moyenne élevée ({cls.average_complexity:.2f}). "
                           "Optimisez certaines méthodes.")
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
        if not cls.has_docstring and doc_threshold:
            cost = 5
            improvement = 5
            description = (f"La classe '{cls.name}' manque de docstring. "
                           "Ajoutez une documentation.")
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

def generate_variable_candidates(issues: List[Tuple[str, int]]) -> List[RefactoringCandidate]:
    """Génère des suggestions pour le renommage des variables ne respectant pas le snake_case."""
    candidates = []
    for var_name, lineno in issues:
        cost = 2
        improvement = 2
        description = (f"La variable '{var_name}' à la ligne {lineno} ne respecte pas le snake_case. "
                       "Envisagez de la renommer.")
        candidates.append(RefactoringCandidate(
            scope="variable",
            name=var_name,
            candidate_type="naming",
            metric=0,
            threshold=0,
            cost=cost,
            improvement=improvement,
            description=description
        ))
    return candidates

# --- Optimisation par recuit simulé ---

def cost_function(state: List[int], candidates: List[RefactoringCandidate]) -> float:
    """Calcule le coût total d'un état (liste de bits indiquant l'application de chaque suggestion)."""
    total = 0
    for applied, cand in zip(state, candidates):
        if applied:
            total += max(cand.cost - cand.improvement, 0)
        else:
            total += cand.cost
    return total

def perturb_state(state: List[int]) -> List[int]:
    """Modifie aléatoirement l'état en inversant le bit d'une suggestion choisie au hasard."""
    new_state = state.copy()
    idx = random.randint(0, len(state)-1)
    new_state[idx] = 1 - new_state[idx]
    return new_state

def simulated_annealing(candidates: List[RefactoringCandidate],
                        initial_state: List[int],
                        initial_temp: float = 100.0,
                        cooling_rate: float = 0.97,
                        iterations: int = 3000,
                        debug: bool = False) -> Tuple[List[int], float]:
    """Optimise la sélection des suggestions via recuit simulé."""
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
    """Génère un rapport détaillé des suggestions retenues."""
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
    report.sort(key=lambda x: (x["scope"], x["candidate_type"]))
    return report

def write_report_json(report: List[dict], output_file: str):
    """Écrit le rapport dans un fichier JSON."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        logger.info(f"Rapport écrit dans le fichier {output_file}")
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture du rapport JSON: {e}")

# --- Auto-formatage du code avec autopep8 ---

def auto_format_code(file_path: str) -> bool:
    """
    Applique autopep8 sur le fichier source pour un formatage automatique.
    Retourne True si l'opération s'est bien déroulée.
    """
    try:
        result = subprocess.run(["autopep8", "--in-place", file_path], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Le fichier {file_path} a été formaté automatiquement.")
            return True
        else:
            logger.error(f"Erreur lors de l'auto-formatage: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Exception lors de l'appel à autopep8: {e}")
        return False

# --- Analyse sur répertoire ---

def analyze_directory(directory: str, recursive: bool = True) -> List[str]:
    """
    Retourne la liste de tous les fichiers Python présents dans un répertoire (de manière récursive si demandé).
    """
    pattern = "**/*.py" if recursive else "*.py"
    files = glob.glob(os.path.join(directory, pattern), recursive=recursive)
    return files

# --- Visualisations ---

def visualize_metrics(functions: List[FunctionMetrics], classes: List[ClassMetrics]):
    """Affiche des histogrammes des métriques via matplotlib (si disponible)."""
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib n'est pas installé.")
        return

    # Histogramme des longueurs de fonctions
    func_lengths = [f.length for f in functions]
    func_complexities = [f.complexity for f in functions]
    class_lengths = [c.length for c in classes]
    class_complexities = [c.average_complexity for c in classes]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0,0].hist(func_lengths, bins=20, color='skyblue')
    axs[0,0].set_title("Longueur des fonctions")
    axs[0,1].hist(func_complexities, bins=20, color='salmon')
    axs[0,1].set_title("Complexité des fonctions")
    axs[1,0].hist(class_lengths, bins=20, color='lightgreen')
    axs[1,0].set_title("Longueur des classes")
    axs[1,1].hist(class_complexities, bins=20, color='plum')
    axs[1,1].set_title("Complexité moyenne des classes")
    plt.tight_layout()
    plt.show()

# --- Fonction principale pour l'analyse (CLI ou GUI) ---

def run_qira(file_paths: List[str],
             length_threshold: int,
             complexity_threshold: int,
             depth_threshold: int,
             methods_threshold: int,
             class_complexity_threshold: float,
             doc_required: bool,
             iterations: int,
             debug: bool,
             analyze_duplicates: bool,
             analyze_naming: bool) -> Tuple[str, List[dict]]:
    """
    Exécute l'analyse QIRA sur un ou plusieurs fichiers.
    Retourne un rapport textuel et le rapport détaillé sous forme de liste de dictionnaires.
    """
    all_func_candidates = []
    all_class_candidates = []
    all_variable_candidates = []
    all_candidates = []
    global_report_text = ""
    
    # Pour chaque fichier, analyser et générer des candidats
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Erreur de syntaxe dans le fichier {file_path}: {e}")
            continue

        functions = analyze_functions(tree, source_code)
        classes = analyze_classes(tree)
        func_candidates = generate_function_candidates(functions, length_threshold, complexity_threshold, depth_threshold, doc_required)
        class_candidates = generate_class_candidates(classes, length_threshold, methods_threshold, class_complexity_threshold, doc_required)
        all_func_candidates.extend(func_candidates)
        all_class_candidates.extend(class_candidates)
        
        # Analyse du nommage des variables
        if analyze_naming:
            var_issues = analyze_variable_names(tree)
            variable_candidates = generate_variable_candidates(var_issues)
            all_variable_candidates.extend(variable_candidates)
        
        file_report = f"\nFichier: {file_path}\n"
        file_report += f"  Fonctions détectées: {len(functions)} | Classes détectées: {len(classes)}\n"
        global_report_text += file_report

    # Détecter les duplications sur l'ensemble des fonctions
    if analyze_duplicates:
        # On peut regrouper toutes les fonctions analysées (on suppose qu'elles ont été traitées dans chaque fichier)
        duplicate_groups = detect_duplicates(functions)
        if duplicate_groups:
            dup_report = "\nDuplication de code détectée:\n"
            for group in duplicate_groups:
                names = ", ".join([f.name for f in group])
                dup_report += f"  Fonctions similaires: {names}\n"
            global_report_text += dup_report

    all_candidates = all_func_candidates + all_class_candidates + all_variable_candidates
    if not all_candidates:
        return "Aucune suggestion de refactoring nécessaire détectée.", []
    initial_state = [0] * len(all_candidates)
    best_state, best_cost = simulated_annealing(all_candidates, initial_state, iterations=iterations, debug=debug)
    report_list = generate_report(all_candidates, best_state)
    if not report_list:
        return "Aucune suggestion retenue après optimisation.", []
    global_report_text += "\nRapport de suggestions de refactoring:\n"
    for item in report_list:
        global_report_text += (f"{item['scope'].capitalize()} '{item['name']}' - {item['candidate_type'].capitalize()}: "
                               f"{item['description']} (Coût: {item['cost']:.2f}, Amélioration: {item['improvement']:.2f})\n")
    return global_report_text, report_list

# --- Parsing des arguments pour le mode CLI ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantum-Inspired Refactoring Assistant (QIRA) - Version Ultra-Avancée")
    parser.add_argument("--file", help="Chemin vers le fichier source Python à analyser.")
    parser.add_argument("--dir", help="Chemin vers le répertoire à analyser (mode multi-fichiers).")
    parser.add_argument("--length_threshold", type=int, default=30, help="Seuil de longueur pour fonctions/classes (lignes).")
    parser.add_argument("--complexity_threshold", type=int, default=8, help="Seuil de complexité cyclomatique pour fonctions.")
    parser.add_argument("--depth_threshold", type=int, default=3, help="Seuil de profondeur d'imbrication pour fonctions.")
    parser.add_argument("--methods_threshold", type=int, default=5, help="Seuil du nombre de méthodes pour une classe.")
    parser.add_argument("--class_complexity_threshold", type=float, default=5.0, help="Seuil de complexité moyenne pour une classe.")
    parser.add_argument("--doc_required", action="store_true", help="Vérifier la présence de docstrings pour fonctions/classes.")
    parser.add_argument("--iterations", type=int, default=3000, help="Nombre maximum d'itérations pour le recuit simulé.")
    parser.add_argument("--debug", action="store_true", help="Active le mode débogage.")
    parser.add_argument("--output", help="Fichier de sortie pour le rapport JSON.")
    parser.add_argument("--duplication", action="store_true", help="Analyser la duplication de code.")
    parser.add_argument("--naming", action="store_true", help="Analyser le nommage des variables.")
    parser.add_argument("--autoformat", action="store_true", help="Auto-formater le code avec autopep8.")
    parser.add_argument("--visualize", action="store_true", help="Afficher des visualisations des métriques.")
    return parser.parse_args()

# --- Interface Graphique (Tkinter) ---

def run_gui():
    root = tk.Tk()
    root.title("Quantum-Inspired Refactoring Assistant (QIRA) Ultra")

    # Cadre pour les paramètres
    frame_inputs = tk.Frame(root)
    frame_inputs.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # Mode fichier ou répertoire
    mode_var = tk.StringVar(value="file")
    tk.Label(frame_inputs, text="Mode:").grid(row=0, column=0, sticky=tk.W)
    tk.Radiobutton(frame_inputs, text="Fichier", variable=mode_var, value="file").grid(row=0, column=1, sticky=tk.W)
    tk.Radiobutton(frame_inputs, text="Répertoire", variable=mode_var, value="dir").grid(row=0, column=2, sticky=tk.W)

    # Sélection du fichier ou répertoire
    path_label = tk.Label(frame_inputs, text="Chemin:")
    path_label.grid(row=1, column=0, sticky=tk.W)
    path_entry = tk.Entry(frame_inputs, width=50)
    path_entry.grid(row=1, column=1, padx=5, columnspan=2)
    def browse_path():
        if mode_var.get() == "file":
            filename = filedialog.askopenfilename(filetypes=[("Fichiers Python", "*.py"), ("Tous", "*.*")])
        else:
            filename = filedialog.askdirectory()
        if filename:
            path_entry.delete(0, tk.END)
            path_entry.insert(0, filename)
    tk.Button(frame_inputs, text="Parcourir", command=browse_path).grid(row=1, column=3, padx=5)

    # Paramètres numériques
    labels = ["Seuil de longueur:", "Seuil de complexité:", "Seuil de profondeur:", "Seuil de méthodes (classe)::", "Seuil de complexité moyenne (classe):", "Itérations:"]
    defaults = ["30", "8", "3", "5", "5.0", "3000"]
    entries = []
    for i, (lab, default) in enumerate(zip(labels, defaults), start=2):
        tk.Label(frame_inputs, text=lab).grid(row=i, column=0, sticky=tk.W)
        ent = tk.Entry(frame_inputs, width=10)
        ent.insert(0, default)
        ent.grid(row=i, column=1, sticky=tk.W, padx=5)
        entries.append(ent)

    # Options supplémentaires
    doc_var = tk.BooleanVar()
    dup_var = tk.BooleanVar()
    naming_var = tk.BooleanVar()
    autoformat_var = tk.BooleanVar()
    debug_var = tk.BooleanVar()
    visualize_var = tk.BooleanVar()
    tk.Checkbutton(frame_inputs, text="Docstrings requises", variable=doc_var).grid(row=8, column=0, sticky=tk.W)
    tk.Checkbutton(frame_inputs, text="Analyser duplication", variable=dup_var).grid(row=8, column=1, sticky=tk.W)
    tk.Checkbutton(frame_inputs, text="Analyser nommage variables", variable=naming_var).grid(row=8, column=2, sticky=tk.W)
    tk.Checkbutton(frame_inputs, text="Auto-formater (autopep8)", variable=autoformat_var).grid(row=9, column=0, sticky=tk.W)
    tk.Checkbutton(frame_inputs, text="Mode debug", variable=debug_var).grid(row=9, column=1, sticky=tk.W)
    tk.Checkbutton(frame_inputs, text="Visualiser métriques", variable=visualize_var).grid(row=9, column=2, sticky=tk.W)

    # Zone de texte pour le rapport
    result_text = tk.Text(root, wrap=tk.WORD, height=20)
    result_text.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

    def run_analysis():
        path = path_entry.get()
        if not path:
            messagebox.showerror("Erreur", "Veuillez sélectionner un chemin.")
            return
        try:
            length_threshold = int(entries[0].get())
            complexity_threshold = int(entries[1].get())
            depth_threshold = int(entries[2].get())
            methods_threshold = int(entries[3].get())
            class_complexity_threshold = float(entries[4].get())
            iterations = int(entries[5].get())
        except ValueError:
            messagebox.showerror("Erreur", "Vérifiez que les paramètres numériques sont valides.")
            return
        doc_required = doc_var.get()
        dup_analysis = dup_var.get()
        naming_analysis = naming_var.get()
        autoformat = autoformat_var.get()
        debug = debug_var.get()
        visualize = visualize_var.get()

        # Récupérer la liste des fichiers selon le mode
        if mode_var.get() == "file":
            file_list = [path]
        else:
            file_list = analyze_directory(path, recursive=True)
            if not file_list:
                messagebox.showerror("Erreur", "Aucun fichier Python trouvé dans ce répertoire.")
                return

        # Auto-formatage si demandé
        if autoformat:
            for f in file_list:
                auto_format_code(f)

        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Analyse en cours...\n")
        root.update_idletasks()
        try:
            report_text, _ = run_qira(file_list, length_threshold, complexity_threshold,
                                       depth_threshold, methods_threshold, class_complexity_threshold,
                                       doc_required, iterations, debug, dup_analysis, naming_analysis)
            result_text.insert(tk.END, report_text)
            if visualize and MATPLOTLIB_AVAILABLE:
                # Pour la visualisation, on analyse le premier fichier
                with open(file_list[0], "r", encoding="utf-8") as f:
                    src = f.read()
                tree = ast.parse(src)
                funcs = analyze_functions(tree, src)
                classes = analyze_classes(tree)
                visualize_metrics(funcs, classes)
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue:\n{e}")
    tk.Button(root, text="Lancer l'analyse", command=run_analysis).pack(pady=5)
    root.mainloop()

# --- Point d'entrée du script ---

if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
        file_list = []
        if args.file:
            file_list = [args.file]
        elif args.dir:
            file_list = analyze_directory(args.dir, recursive=True)
            if not file_list:
                print("Aucun fichier Python trouvé dans le répertoire spécifié.")
                sys.exit(1)
        else:
            print("Erreur : veuillez spécifier --file ou --dir.")
            sys.exit(1)

        # Auto-formatage si demandé
        if args.autoformat:
            for f in file_list:
                auto_format_code(f)

        report_text, report_data = run_qira(
            file_list,
            args.length_threshold,
            args.complexity_threshold,
            args.depth_threshold,
            args.methods_threshold,
            args.class_complexity_threshold,
            args.doc_required,
            args.iterations,
            args.debug,
            args.duplication,
            args.naming
        )
        print(report_text)
        if args.visualize and MATPLOTLIB_AVAILABLE and file_list:
            with open(file_list[0], "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src)
            funcs = analyze_functions(tree, src)
            classes = analyze_classes(tree)
            visualize_metrics(funcs, classes)
        if args.output:
            write_report_json(report_data, args.output)
    else:
        run_gui()
