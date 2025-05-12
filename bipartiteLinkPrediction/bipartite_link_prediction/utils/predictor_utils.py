from collections import defaultdict
from typing import Tuple, List, Any

from bipartite_link_prediction.bipartite_network import BipartiteNetwork


def create_train_top_adjacency_list(bipartite_network: BipartiteNetwork,
                                    test_edges: List[Tuple[int, int]]) -> defaultdict[Any, list]:
    """
    Creates training adjacency lists that exclude edges designated for testing.

    :param bipartite_network:
    :param test_edges: List of edges to be used as test set.
    :return: Training adjacency lists for top and bottom nodes.
    """

    test_edges_set = set(test_edges)
    train_top_adj = defaultdict(list)
    for u, neighbors in bipartite_network.adjacency_list_top.items():
        for v in neighbors:
            if (u, v) not in test_edges_set:
                train_top_adj[u].append(v)

    return train_top_adj


def create_train_bottom_adjacency_list(bipartite_network: BipartiteNetwork,
                                       test_edges: List[Tuple[int, int]]) -> defaultdict[Any, list]:
    """
    Creates training adjacency lists that exclude edges designated for testing.

    :param bipartite_network:
    :param test_edges: List of edges to be used as test set.
    :return: Training adjacency lists for top and bottom nodes.
    """

    test_edges_set = set(test_edges)
    train_bottom_adj = defaultdict(list)
    for v, neighbors in bipartite_network.adjacency_list_bottom.items():
        for u in neighbors:
            if (u, v) not in test_edges_set:
                train_bottom_adj[v].append(u)

    return train_bottom_adj
