from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

Relationships = Dict[str, Dict[str, List[str]]]
Edge = Tuple[str, str, str]


def _generate_name(rng: random.Random, used: set[str]) -> str:
    while True:
        name = "".join(rng.choices(string.ascii_uppercase, k=2))
        if name not in used:
            used.add(name)
            return name


def _add_relation(relationships: MutableMapping[str, Dict[str, List[str]]], source: str, relation: str, target: str) -> None:
    relationships.setdefault(source, {}).setdefault(relation, [])
    if target not in relationships[source][relation]:
        relationships[source][relation].append(target)


def _merge_relationships(left: Relationships, right: Relationships) -> Relationships:
    merged: Relationships = {person: {rel: list(targets) for rel, targets in rels.items()} for person, rels in left.items()}
    for person, rels in right.items():
        for relation, targets in rels.items():
            for target in targets:
                _add_relation(merged, person, relation, target)
    return merged


def create_nuclear_family(
    num_children: int = 2,
    child_names: Optional[Sequence[str]] = None,
    *,
    rng: Optional[random.Random] = None,
    used_names: Optional[set[str]] = None,
) -> Tuple[Relationships, Dict[str, str]]:
    rng = rng or random.Random()
    used_names = used_names if used_names is not None else set(child_names or [])
    child_names = list(child_names or [])
    parent1 = _generate_name(rng, used_names)
    parent2 = _generate_name(rng, used_names)
    children = list(child_names)
    while len(children) < num_children:
        children.append(_generate_name(rng, used_names))

    relationships: Relationships = {}
    _add_relation(relationships, parent1, "SPOUSE_OF", parent2)
    _add_relation(relationships, parent2, "SPOUSE_OF", parent1)
    for child in children:
        _add_relation(relationships, parent1, "PARENT_OF", child)
        _add_relation(relationships, parent2, "PARENT_OF", child)
        _add_relation(relationships, child, "CHILD_OF", parent1)
        _add_relation(relationships, child, "CHILD_OF", parent2)
        for sibling in children:
            if sibling != child:
                _add_relation(relationships, child, "SIBLING_OF", sibling)

    roles = {parent1: "parent", parent2: "parent"}
    roles.update({child: "child" for child in children})
    return relationships, roles


def create_extended_family_tree(
    base_num_children: int = 2,
    grandparent_num_children: int = 2,
    *,
    seed: Optional[int] = None,
) -> Relationships:
    rng = random.Random(seed)
    used_names: set[str] = set()
    base_relationships, base_roles = create_nuclear_family(
        base_num_children,
        rng=rng,
        used_names=used_names,
    )
    parent_names = [name for name, role in base_roles.items() if role == "parent"]
    relationships = base_relationships
    for parent_name in parent_names:
        grandparent_relationships, _roles = create_nuclear_family(
            grandparent_num_children,
            child_names=[parent_name],
            rng=rng,
            used_names=used_names,
        )
        relationships = _merge_relationships(relationships, grandparent_relationships)
    return infer_grandparent_edges(relationships)


def infer_grandparent_edges(relationships: Relationships) -> Relationships:
    inferred: Relationships = {}
    for person, rels in relationships.items():
        for child in rels.get("PARENT_OF", []):
            for grandchild in relationships.get(child, {}).get("PARENT_OF", []):
                _add_relation(inferred, person, "GRANDPARENT_OF", grandchild)
                _add_relation(inferred, grandchild, "GRANDCHILD_OF", person)
    return _merge_relationships(relationships, inferred)


def relationship_edges(relationships: Relationships) -> List[Edge]:
    edges: List[Edge] = []
    for source, rels in relationships.items():
        for relation, targets in rels.items():
            for target in targets:
                edges.append((source, relation, target))
    return edges


def relationship_texts(relationships: Relationships) -> List[str]:
    return [f"{source} {relation} {target}" for source, relation, target in relationship_edges(relationships)]


def generate_random_walk(
    relationships: Relationships,
    *,
    walk_length: int = 20,
    rng: Optional[random.Random] = None,
) -> str:
    rng = rng or random.Random()
    current = rng.choice(list(relationships))
    parts = [current]
    for _ in range(walk_length):
        rels = relationships.get(current, {})
        options = [(relation, target) for relation, targets in rels.items() for target in targets]
        if not options:
            break
        relation, target = rng.choice(options)
        parts.extend([relation, target])
        current = target
    return " ".join(parts)


def make_corpus(
    path: Union[str, Path],
    *,
    num_trees: int = 100,
    walks_per_tree: int = 10,
    walk_length: int = 20,
    base_num_children: int = 2,
    grandparent_num_children: int = 2,
    seed: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with path.open("w") as handle:
        for _ in range(num_trees):
            tree = create_extended_family_tree(
                base_num_children=base_num_children,
                grandparent_num_children=grandparent_num_children,
                seed=rng.randint(0, 10_000_000),
            )
            for _ in range(walks_per_tree):
                handle.write(generate_random_walk(tree, walk_length=walk_length, rng=rng) + "\n")
    return path


def semantic_qa_pairs(relationships: Relationships, *, max_questions: Optional[int] = None) -> List[dict]:
    questions: List[dict] = []
    for person, rels in relationships.items():
        for spouse in rels.get("SPOUSE_OF", []):
            questions.append({"question": f"Who is {person} married to?", "answer": spouse, "relation": "SPOUSE_OF"})
        for child in rels.get("PARENT_OF", []):
            questions.append({"question": f"Who is a child of {person}?", "answer": child, "relation": "PARENT_OF"})
            questions.append({"question": f"Who is a parent of {child}?", "answer": person, "relation": "CHILD_OF"})
        for sibling in rels.get("SIBLING_OF", []):
            questions.append({"question": f"Who is a sibling of {person}?", "answer": sibling, "relation": "SIBLING_OF"})
        for grandchild in rels.get("GRANDPARENT_OF", []):
            questions.append({"question": f"Who is a grandchild of {person}?", "answer": grandchild, "relation": "GRANDPARENT_OF"})
            questions.append({"question": f"Who is a grandparent of {grandchild}?", "answer": person, "relation": "GRANDCHILD_OF"})
    return questions[:max_questions] if max_questions is not None else questions


def episodic_recall_items(walks: Iterable[str], *, cue_tokens: int = 5) -> List[dict]:
    items = []
    for walk in walks:
        tokens = walk.split()
        if len(tokens) <= cue_tokens:
            continue
        cue = " ".join(tokens[:cue_tokens])
        target = " ".join(tokens[cue_tokens:])
        items.append({"cue": cue, "target": target, "text": walk})
    return items

