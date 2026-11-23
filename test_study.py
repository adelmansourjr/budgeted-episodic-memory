"""Audit exact routing, observability, accounting, and corpus split invariants."""
from fractions import Fraction as F
from itertools import combinations
import json
import unittest

import numpy as np

from routing_bound import Node, adaptive_coverage, frontier, full_tree
from study import DATA, Index, ReadLedger, View, informative, inputs, load, make_split, oracle, select, words


def brute_force(tree, budget):
    paths = []
    costs = []
    def enumerate_paths(node, route):
        i = len(costs)
        costs.append(node.cost)
        if node.children:
            for child in node.children:
                enumerate_paths(child, route | {i})
        else:
            paths.append((route | {i}, node.probability))
    enumerate_paths(tree, set())
    best = F()
    for size in range(len(costs) + 1):
        for chosen in combinations(range(len(costs)), size):
            if sum(costs[i] for i in chosen) <= budget:
                bought = set(chosen)
                best = max(best, sum((p for route,p in paths if route <= bought), F()))
    return best


class ResearchChecks(unittest.TestCase):
    def test_weighted_routing_matches_exhaustive_subsets(self):
        tree = Node(2, children=(
            Node(2, children=(Node(1,F(1,4)),Node(2,F(1,8)))),
            Node(1, children=(Node(1,F(1,2)),Node(3,F(1,8)))),
        ))
        total = 12
        self.assertEqual(frontier(tree,total), [brute_force(tree,b) for b in range(total+1)])

    def test_uniform_closed_form_and_adaptive_budget(self):
        for depth in range(1, 4):
            for branching in range(1, 5):
                budget = depth + 1
                tree = full_tree(depth,branching)
                self.assertEqual(frontier(tree,budget)[budget], F(1,branching**depth))
                self.assertEqual(adaptive_coverage(tree,budget), 1)
                self.assertEqual(adaptive_coverage(tree,budget-1), 0)
        self.assertEqual(frontier(full_tree(1,16),6)[6],F(5,16))

    def test_budget_and_upper_bound(self):
        views = (View(("dot",),("loud",),3),View(("cat",),("quiet",),5),View((),(),1))
        tails = (("toy","rabbit"),("three","more","words","here"),())
        ledger=ReadLedger(views,"words",5,tails,[])
        self.assertEqual(ledger.read(0),("toy","rabbit"))
        with self.assertRaises(ValueError):
            ledger.read(1)
        with self.assertRaises(ValueError):
            ledger.read(0)
        self.assertEqual(ledger.read(2),())
        self.assertEqual(ledger.words_spent,4)
        self.assertEqual(oracle((True,True,False), views,"words",5),1)

    def test_unread_payload_is_invisible_and_reveal_changes_choice(self):
        views = (
            View(("station",),("station",),2),
            View(("alpha",),("news",),1),
            View(("beta",),("news",),1),
        )
        sequences = []
        for term in ("alpha","beta"):
            index = Index(["station"],views)
            ledger=ReadLedger(views,"reads",2,((term,),(),()),[])
            select(index,ledger,"reveal_bridge",{"weight":.75,"title_share":.67})
            self.assertEqual(ledger.selected[0],0)
            sequences.append(ledger.selected)
        self.assertEqual(sequences,[[0,1],[0,2]])

    @unittest.skipUnless(DATA.exists(), "Download data to audit the corpus split")
    def test_split_disjoint_for_support_titles_and_source_ids(self):
        rows = load()
        groups = {}
        splits = {}
        for row, (split,group) in zip(rows,make_split(rows)):
            self.assertIn(split, ("test","development","topic_stress"))
            for key in [
                *(("source",str(x["id"])) for x in row["question_decomposition"]),
                *(("title"," ".join(words(x["title"]))) for x in row["paragraphs"] if x["is_supporting"])
            ]:
                self.assertEqual(groups.setdefault(key,group),group)
                self.assertEqual(splits.setdefault(key,split),split)
        row = rows[0]
        q, views, tails, _, ids = inputs(row,24)
        remap = {p["idx"]:p for p in row["paragraphs"]}
        for doc, ix, tail in zip(views,ids,tails):
            tok = words(remap[ix]["paragraph_text"])
            self.assertEqual(doc.preview,tuple(tok[:24]))
            self.assertEqual(tail,tuple(tok[24:]))


if __name__ == "__main__":
    unittest.main()
