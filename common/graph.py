from itertools import combinations
from typing import List, Dict, Set
from collections import defaultdict, deque


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """Topological sort for a DAG

    Args:
        `graph` (Dict[str, List[str]]): A dictionary representing the adjacency list of the graph.

    Returns:
        List (List[str]): A list of nodes in topologically sorted order.
    """
    in_degree = defaultdict(int)
    for node in graph:
        in_degree[node]
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    queue = deque([node for node in in_degree if in_degree[node] == 0])

    sorted_order = []

    while queue:
        current = queue.popleft()
        sorted_order.append(current)

        for neighbor in graph.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_order) == len(in_degree):
        return sorted_order
    else:
        assert False, "Error: Graph contains a cycle, topological sort not possible."


def invert_graph(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Invert a graph.

    Args:
        `graph` (Dict[str, List[str]]): A dictionary representing the adjacency list of the graph.

    Returns:
        Dict (Dict[str, List[str]]): Inverted graph.
    """
    inverted = {node: [] for node in graph}

    for node, children in graph.items():
        for child in children:
            inverted[child].append(node)

    return inverted


def interventionally_meaningful_subsets(graph) -> List[Set[str]]:
    """Return all subsets :math:`s` satisfying:
    1) No nodes with zero children appear in :math:`s`.
    2) If :math:`p` and :math:`c` are both in :math:`s` and :math:`p` is a parent of :math:`c`,
        then :math:`p` and :math:`c` share at least one child in common.
    3) :math:`|s|>=1`.

    Args:
        `graph` (Dict[str, List[str]]): A dictionary representing the adjacency list of the graph.

    Returns:
        List (List[Set[str]]): Interventionally meaningful subsets
    """
    nodes_with_children = [
        node for node, children in graph.items() if children
    ]

    parents_of = {}
    for node in graph:
        parents_of[node] = []
    for p, children in graph.items():
        for c in children:
            parents_of[c].append(p)

    def share_at_least_one_child(p, c):
        return len(set(graph[p]) & set(graph[c])) > 0

    def is_valid_subset(s):
        s_set = set(s)
        for c in s:
            for p in parents_of[c]:
                if p in s_set:
                    if not share_at_least_one_child(p, c):
                        return False
        return True

    valid_subsets = []
    n = len(nodes_with_children)
    for r in range(1, n + 1):
        for combo in combinations(nodes_with_children, r):
            if is_valid_subset(combo):
                valid_subsets.append(set(combo))

    return valid_subsets
