import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _gen_name(rng: random.Random) -> str:
    letters = string.ascii_uppercase
    return rng.choice(letters) + rng.choice(letters)


def generate_named_square_loop(rng: random.Random, size: int) -> str:
    # Coordinates around a square loop starting at (0,0)
    coords = [(0, 0)]
    x, y = 0, 0
    for _ in range(size):
        y += 1
        coords.append((x, y))
    for _ in range(size):
        x += 1
        coords.append((x, y))
    for _ in range(size):
        y -= 1
        coords.append((x, y))
    for _ in range(size):
        x -= 1
        coords.append((x, y))

    # Assign names to unique coords
    coord_to_name: Dict[Tuple[int, int], str] = {}
    for c in coords:
        if c not in coord_to_name:
            coord_to_name[c] = _gen_name(rng)

    parts: List[str] = []
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        name0 = coord_to_name[(x0, y0)]
        if x1 == x0 and y1 == y0 + 1:
            d = "N"
        elif x1 == x0 and y1 == y0 - 1:
            d = "S"
        elif y1 == y0 and x1 == x0 + 1:
            d = "E"
        elif y1 == y0 and x1 == x0 - 1:
            d = "W"
        else:
            d = "?"
        parts.append(name0)
        parts.append(d)
    parts.append(coord_to_name[coords[-1]])
    return " ".join(parts)


def generate_named_square_loop_with_positions(rng: random.Random, size: int):
    coords = [(0, 0)]
    x, y = 0, 0
    for _ in range(size):
        y += 1
        coords.append((x, y))
    for _ in range(size):
        x += 1
        coords.append((x, y))
    for _ in range(size):
        y -= 1
        coords.append((x, y))
    for _ in range(size):
        x -= 1
        coords.append((x, y))

    coord_to_name: Dict[Tuple[int, int], str] = {}
    for c in coords:
        if c not in coord_to_name:
            coord_to_name[c] = _gen_name(rng)

    parts: List[str] = []
    name_order: List[Tuple[str, Tuple[int, int]]] = []
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        name0 = coord_to_name[(x0, y0)]
        if x1 == x0 and y1 == y0 + 1:
            d = "N"
        elif x1 == x0 and y1 == y0 - 1:
            d = "S"
        elif y1 == y0 and x1 == x0 + 1:
            d = "E"
        elif y1 == y0 and x1 == x0 - 1:
            d = "W"
        else:
            d = "?"
        parts.append(name0)
        name_order.append((name0, (x0, y0)))
        parts.append(d)
    parts.append(coord_to_name[coords[-1]])
    name_order.append((coord_to_name[coords[-1]], coords[-1]))
    text = " ".join(parts)
    return text, name_order, coords


def make_pretrain_corpus(path: str, num_graphs: int = 200, loops_per_graph: int = 20, min_size: int = 1, max_size: int = 3, seed: int = 0):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(num_graphs):
            for _ in range(loops_per_graph):
                s = rng.randint(min_size, max_size)
                f.write(generate_named_square_loop(rng, s) + "\n")


def make_icl_context_and_probe(rng: random.Random, size: int = 3, k_context: int = 4):
    # Build multiple loops using a shared name mapping (same graph), then a probe missing last direction
    full = generate_named_square_loop(rng, size)
    names = full.split()
    # Synthesize k-1 additional loops by regenerating with same RNG state for names
    ctx_lines = [full]
    for _ in range(k_context - 1):
        ctx_lines.append(generate_named_square_loop(rng, size))
    # Build a probe that omits the final direction before last name
    parts = full.split()
    # parts: NAME DIR NAME DIR ... NAME
    # Final move is the direction at index -2
    target_dir = parts[-2]
    probe = " ".join(parts[:-2])  # up to previous NAME
    return ctx_lines, probe, target_dir


DIRECTION_DELTAS = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
    "NORTH": (0, 1),
    "EAST": (1, 0),
    "SOUTH": (0, -1),
    "WEST": (-1, 0),
}


@dataclass(frozen=True)
class SpatialGraph:
    adjacency: Dict[str, List[Tuple[str, str]]]
    positions: Dict[str, Tuple[int, int]]


def make_grid_graph(
    *,
    size: int = 3,
    names: Optional[Sequence[str]] = None,
    seed: int = 0,
    long_direction_names: bool = False,
) -> SpatialGraph:
    if size < 2:
        raise ValueError("size must be at least 2.")
    rng = random.Random(seed)
    if names is None:
        names = []
        used = set()
        while len(names) < size * size:
            name = _gen_name(rng)
            if name not in used:
                used.add(name)
                names.append(name)
    if len(names) != size * size:
        raise ValueError("names must contain size * size items.")

    directions = {
        "N" if not long_direction_names else "NORTH": (0, 1),
        "E" if not long_direction_names else "EAST": (1, 0),
        "S" if not long_direction_names else "SOUTH": (0, -1),
        "W" if not long_direction_names else "WEST": (-1, 0),
    }
    positions: Dict[str, Tuple[int, int]] = {}
    coord_to_name: Dict[Tuple[int, int], str] = {}
    for y in range(size):
        for x in range(size):
            name = names[y * size + x]
            positions[name] = (x, y)
            coord_to_name[(x, y)] = name

    adjacency: Dict[str, List[Tuple[str, str]]] = {name: [] for name in names}
    for name, (x, y) in positions.items():
        for direction, (dx, dy) in directions.items():
            target_coord = (x + dx, y + dy)
            if target_coord in coord_to_name:
                adjacency[name].append((direction, coord_to_name[target_coord]))
    return SpatialGraph(adjacency=adjacency, positions=positions)


def generate_random_walk(
    graph: SpatialGraph,
    *,
    walk_length: int = 20,
    rng: Optional[random.Random] = None,
) -> str:
    rng = rng or random.Random()
    current = rng.choice(list(graph.adjacency))
    parts = [current]
    for _ in range(walk_length):
        options = graph.adjacency.get(current, [])
        if not options:
            break
        direction, target = rng.choice(options)
        parts.extend([direction, target])
        current = target
    return " ".join(parts)


def make_grid_corpus(
    path: Union[str, Path],
    *,
    num_graphs: int = 100,
    walks_per_graph: int = 10,
    walk_length: int = 20,
    grid_size: int = 3,
    seed: int = 0,
    long_direction_names: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with path.open("w") as handle:
        for _ in range(num_graphs):
            graph = make_grid_graph(
                size=grid_size,
                seed=rng.randint(0, 10_000_000),
                long_direction_names=long_direction_names,
            )
            for _ in range(walks_per_graph):
                handle.write(generate_random_walk(graph, walk_length=walk_length, rng=rng) + "\n")
    return path


def parse_path_positions(text: str, start: Tuple[int, int] = (0, 0)) -> List[Tuple[str, Tuple[int, int]]]:
    tokens = text.split()
    if not tokens:
        return []
    x, y = start
    positions = [(tokens[0], (x, y))]
    index = 1
    while index + 1 < len(tokens):
        direction = tokens[index]
        node = tokens[index + 1]
        dx, dy = DIRECTION_DELTAS.get(direction, (0, 0))
        x, y = x + dx, y + dy
        positions.append((node, (x, y)))
        index += 2
    return positions


def is_valid_path(text: str, graph: SpatialGraph) -> bool:
    tokens = text.split()
    if len(tokens) < 3:
        return True
    current = tokens[0]
    index = 1
    while index + 1 < len(tokens):
        direction = tokens[index]
        target = tokens[index + 1]
        if (direction, target) not in graph.adjacency.get(current, []):
            return False
        current = target
        index += 2
    return True


def episodic_recall_items(paths: Iterable[str], *, cue_tokens: int = 5) -> List[dict]:
    items = []
    for path in paths:
        tokens = path.split()
        if len(tokens) <= cue_tokens:
            continue
        cue = " ".join(tokens[:cue_tokens])
        target = " ".join(tokens[cue_tokens:])
        items.append({"cue": cue, "target": target, "text": path})
    return items

