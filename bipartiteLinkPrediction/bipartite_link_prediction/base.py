from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional

import numpy as np

from .bipartite_network import BipartiteNetwork


class BaseLinkPredictor(ABC):
    def __init__(self):
        self._is_data_prepared = False


    @abstractmethod
    def prepare_data(self, bipartite_network: BipartiteNetwork, test_edges: Optional[List[Tuple[int, int]]] = None) -> Any:
        """
        Prepare the training data structure necessary for the predictor.

        :param bipartite_network: An instance of BipartiteNetwork containing the network data.
        :param test_edges: Optional list of edges to exclude from training (default=None).
        :return: The data structure needed for fitting.
        """
        pass

    @abstractmethod
    def fit(self, data: Any) -> None:
        """
        Train the predictor model on the provided data structure.

        :param data: The training data structure prepared for the predictor.
        """
        if not self._is_data_prepared:
            raise RuntimeError("prepare_data must be called before fit.")

    @abstractmethod
    def predict_links(self, node_pairs: List[Tuple[int, int]], trial_num: Optional[int] = None) -> np.ndarray[Any, np.dtype[float]]:
        """
        Predict links based on the trained model.

        :return: List of predicted links as tuples of node indices.
        """
        pass


