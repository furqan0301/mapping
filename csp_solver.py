from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class SolveStats:
    success: bool
    backtracks: int
    assignments: int
    elapsed_ms: int


def _is_consistent(var: str, value: str, assignment: Dict[str, str], neighbors: Dict[str, List[str]]) -> bool:
    for nb in neighbors.get(var, []):
        if assignment.get(nb) == value:
            return False
    return True


def _select_unassigned_var(variables: List[str],
                          assignment: Dict[str, str],
                          domains: Dict[str, List[str]],
                          neighbors: Dict[str, List[str]]) -> str:
    # MRV + degree heuristic
    unassigned = [v for v in variables if v not in assignment]

    def key(v: str):
        return (len(domains[v]), -len(neighbors.get(v, [])))

    return min(unassigned, key=key)


def _forward_check(var: str, value: str,
                   assignment: Dict[str, str],
                   domains: Dict[str, List[str]],
                   neighbors: Dict[str, List[str]]) -> Optional[List[Tuple[str, str]]]:
    pruned: List[Tuple[str, str]] = []
    for nb in neighbors.get(var, []):
        if nb in assignment:
            continue
        if value in domains[nb]:
            domains[nb].remove(value)
            pruned.append((nb, value))
            if len(domains[nb]) == 0:
                # restore and fail
                for v, val in pruned:
                    domains[v].append(val)
                return None
    return pruned


def _restore(domains: Dict[str, List[str]], pruned: List[Tuple[str, str]]) -> None:
    for v, val in pruned:
        if val not in domains[v]:
            domains[v].append(val)


def solve_map_coloring(countries: List[str],
                       neighbors: Dict[str, List[str]],
                       colors: List[str]):
    start = time.time()

    variables = list(countries)
    domains = {v: list(colors) for v in variables}
    assignment: Dict[str, str] = {}

    backtracks = 0
    assignments = 0

    def backtrack():
        nonlocal backtracks, assignments

        if len(assignment) == len(variables):
            return dict(assignment)

        var = _select_unassigned_var(variables, assignment, domains, neighbors)

        # Least constraining value (simple)
        def lcv_score(val: str) -> int:
            score = 0
            for nb in neighbors.get(var, []):
                if nb not in assignment and val in domains[nb]:
                    score += 1
            return score

        for value in sorted(domains[var], key=lcv_score):
            if _is_consistent(var, value, assignment, neighbors):
                assignment[var] = value
                assignments += 1

                pruned = _forward_check(var, value, assignment, domains, neighbors)
                if pruned is not None:
                    result = backtrack()
                    if result is not None:
                        return result
                    _restore(domains, pruned)

                assignment.pop(var, None)

        backtracks += 1
        return None

    solution = backtrack()
    elapsed_ms = int((time.time() - start) * 1000)

    stats = SolveStats(
        success=solution is not None,
        backtracks=backtracks,
        assignments=assignments,
        elapsed_ms=elapsed_ms
    )
    return solution, stats
