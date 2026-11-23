"""Exact best fixed-batch evidence coverage over a tree of possible routes."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Node:
    cost: int
    probability: Fraction = Fraction(0)
    children: tuple["Node", ...] = ()


def frontier(node: Node, budget: int) -> list[Fraction]:
    """For every read budget, return the best mass for fully covered leaf paths.

    Each node is a separate read. Leaf masses are unconditional probabilities.
    Internal nodes reveal the identity of the next branch only after purchase.
    """
    if budget < 0:
        raise ValueError("Negative budget")
    if node.cost < 1 or node.probability < 0 or (node.children and node.probability):
        raise ValueError("Invalid routing tree")
    if node.cost > budget:
        return [Fraction()] * (budget + 1)
    if not node.children:
        return [Fraction()] * node.cost + [node.probability] * (budget + 1 - node.cost)
    remaining = budget - node.cost
    merged = [Fraction()] * (remaining + 1)
    for child in node.children:
        below = frontier(child, remaining)
        merged = [max(merged[k] + below[b-k] for k in range(b+1)) for b in range(remaining + 1)]
    return [Fraction()] * node.cost + merged


def adaptive_coverage(node: Node, budget: int) -> Fraction:
    """Follow the revealed route. The result assumes every internal read is exact."""
    left = budget - node.cost
    if left < 0:
        return Fraction()
    if not node.children:
        return node.probability
    return sum((adaptive_coverage(child, left) for child in node.children), Fraction())


def full_tree(depth, branches):
    if depth < 0 or branches < 1:
        raise ValueError("Expected nonnegative depth and positive branching")
    if depth == 0:
        return Node(1, Fraction(1))
    def subtree(d):
        if d == 0:
            return Node(1, Fraction(1, branches**depth))
        child = subtree(d-1)
        return Node(1, children=(child,) * branches)
    return subtree(depth)


if __name__ == "__main__":
    for depth, branches in ((1, 16), (2, 4), (3, 4)):
        tree = full_tree(depth, branches)
        budget = depth + 1
        print(f"depth={depth}, branches={branches}, budget={budget}, "
              f"fixed={frontier(tree, budget)[budget]}, adaptive={adaptive_coverage(tree, budget)}")
