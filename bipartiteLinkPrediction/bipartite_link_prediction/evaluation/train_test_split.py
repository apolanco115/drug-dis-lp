from __future__ import annotations

from typing import List, Tuple, Optional, Any
import random
import numpy as np
from numpy import ndarray, dtype

from bipartite_link_prediction.bipartite_network import BipartiteNetwork


class TrainTestSplit:
    def __init__(self, bipartite_network: BipartiteNetwork, num_trials: int = 10, test_size: float = 0.1, test_edges_file: Optional[str] = None, non_edges_file: Optional[str] = None,
                 num_non_edges: Optional[int] = None):
        """
        Initializes the TrainTestSplit class with a BipartiteNetwork object, the number of cross-validation folds,
        the test set size, and an optional file containing test edges.

        :param bipartite_network: An instance of BipartiteNetwork.
        :param num_trials: The number of test trials.
        :param test_size: Fraction of edges to include in the test set.
        :param test_edges_file: Optional path to the file containing test edges.
        :param non_edges_file: Optional path to the file containing non edges.
        """
        self.bipartite_network = bipartite_network
        if num_non_edges is not None:
            self.num_non_edges = int(num_non_edges)
        else:
            num_top = len(self.bipartite_network.top_nodes)
            num_bottom = len(self.bipartite_network.bottom_nodes)
            self.num_non_edges = num_top * num_bottom - self.bipartite_network.graph.num_edges()
        self.num_trials = num_trials
        self.test_size = test_size
        self.test_edges_file = test_edges_file
        self.non_edges_file = non_edges_file

        if self.test_edges_file:
            self.test_edges = self._load_test_edges()
        else:
            raise ValueError("Test edges file not provided. Please provide a test edge file.")

        self.edge_lab = [1] * len(self.test_edges)

        if self.non_edges_file:
            self.non_edges = self._load_non_edges()
        else:
            raise ValueError("Test edges file not provided. Please provide a test edge file.")

        self.non_edge_lab = [0] * len(self.non_edges)

    def _load_test_edges(self) -> List[Tuple[int, int]]:
        """
        Loads test edges from a file. The file should contain one edge per line,
        with node indices separated by a comma. Each edge should be in the format:
        (top_node, bottom_node).

        :param test_edges_file: Path to the file containing test edges.
        :return: List of test edges as pairs of node indices.
        """
        print('loading test edges')
        with open(self.test_edges_file, 'rt') as file:
            test_edges = [tuple(map(int, line.strip().split(','))) for line in file]

        return test_edges

    def _load_non_edges(self) -> list[tuple[int, int]]:
        """
        Loads non-edges from a file. The file should contain one edge per line,
        with node indices separated by a comma. Each edge should be in the format:
        (top_node, bottom_node).

        :return: List of non-edges as pairs of node indices.
        """
        print('loading non-edges')
        with open(self.non_edges_file, 'rt') as file:
            non_edges = [tuple(map(int, line.strip().split(','))) for line in file]

        return non_edges[:self.num_non_edges]

    def get_trial_data(self, trial_index: int) -> Tuple[List, np.ndarray, int]:
        """
        Returns the data for a single trial, including test edges, non-edges, and labels.
        :param trial_index: The index of the trial.
        :return: Tuple containing the test data, labels
        """
        num_test_edges_per_trial = int(self.test_size * self.bipartite_network.graph.num_edges())
        start_index = trial_index * num_test_edges_per_trial
        end_index = start_index + num_test_edges_per_trial
        trial_test_edges = self.test_edges[start_index:end_index]
        trial_num_edges = len(trial_test_edges)
        trial_test_labels = self.edge_lab[start_index:end_index]

        trial_data = trial_test_edges + self.non_edges
        trial_labels = np.concatenate((trial_test_labels, self.non_edge_lab))

        return trial_data, trial_labels, trial_num_edges

