from typing import Any, Optional, List, Tuple

from bipartite_link_prediction.base import BaseLinkPredictor
from bipartite_link_prediction.bipartite_network import BipartiteNetwork

import graph_tool.all as gt
import numpy as np
import scipy.special as spsp
import time


class SBMPredictor(BaseLinkPredictor):
    def __init__(self, eq_steps=10000, sweeps_per_itr: int = 10, mcmc_itr: int = 1000, write_preds: bool = False,
                 save_file_path: Optional[str] = "./bipartite_link_prediction/data/results/") -> None:
        """
        Initialize the SBM predictor.

        Args:
        eq_steps (int): The number of equilibration steps
        sweeps_per_itr (int): The number of sweeps per iteration (default=10).
        mcmc_itr (int): The number of MCMC iterations (default=1000).
        write_preds (bool): Whether to write the predictions to a file (default=False).
        save_file_path (Optional[str]): The path to save the predictions (default=None).

        Returns:
        None
        """
        self._is_data_prepared = False
        self.eq_steps = eq_steps
        self.sweeps_per_itr = sweeps_per_itr
        self.mcmc_itr = mcmc_itr

        self.probs = None
        self.node_pairs = None
        self.counter = None
        self.state = None
        self.node_types = None
        self.graph = None
        self.write_preds = write_preds
        self.save_file_path = save_file_path

    def prepare_data(self, bipartite_network: BipartiteNetwork,
                     test_edges: Optional[List[Tuple[int, int]]] = None) -> tuple[gt.Graph, np.ndarray]:
        """
        Prepare the training data structure necessary for the predictor.

        Args:
        bipartite_network (BipartiteNetwork): The bipartite network object.
        test_edges (Optional[List[Tuple[int, int]]]): A list of test edges to exclude from the training data.

        Returns:
        tuple[gt.Graph, np.ndarray]: The data structure necessary for fitting the predictor.
        """
        test_edges_set = list(map(set, test_edges))
        train_graphview = gt.GraphView(bipartite_network.graph,
                                       efilt=lambda e: set(e) not in test_edges_set)
        train_graph = gt.Graph(train_graphview, prune=True)
        self._is_data_prepared = True
        return train_graph, bipartite_network.node_type

    def fit(self, data: Any) -> None:
        """
        Fit the model to the training data.

        Args:
        data (Any): The data structure necessary for fitting the predictor.

        Returns:
        None
        """
        if not self._is_data_prepared:
            raise RuntimeError("prepare_data must be called before fit.")

        self.graph, self.node_types = data
        print(f'equilibrating for {self.eq_steps} steps')
        bstate = gt.minimize_blockmodel_dl(self.graph,
                                               state_args=dict(deg_corr=True,
                                                               pclabel=self.node_types))
        self.state = gt.MeasuredBlockState(self.graph, n=self.graph.new_ep("int", val=1),
                                       x=self.graph.new_ep("int", val=1),
                                       bstate=bstate, state_args=dict(deg_corr=True,
                                                               pclabel=self.node_types), nested=False)
        gt.mcmc_equilibrate(self.state, force_niter=self.eq_steps, mcmc_args=dict(niter=1))
        self._is_data_prepared = False

    def collect_edge_probas(self, s: gt.MCMCState) -> None:
        """
        Collect the edge probabilities for the current state.

        Args:
        s (gt.MCMCState): The current state of the Markov chain.

        Returns:
        None
        """
        print('iteration:', self.counter)
        self.pair_probas[self.counter] = s.get_edges_prob(self.node_pairs, entropy_args=dict(partition_dl = False))
        self.counter += 1

    def predict_links(self,
                      node_pairs: List[Tuple[int, int]],
                      labels: Optional[np.ndarray] = None,
                      trial_num: Optional[int] = None) -> Optional[np.ndarray]:

        """
        Computes the prediction scores for given node pairs.

        Args:
        node_pairs (List[Tuple[int, int]]): A list of tuples representing the node pairs for which to predict links.
        labels (Optional[np.ndarray]): An array of integer ground truth labels for the node pairs.
        trial_num (Optional[int]): The trial number, used for naming output files.

        Returns:
        Optional[np.ndarray]: An array of predicted scores if write_preds is False;
                              Otherwise, None is returned as scores are written to a file.
        """

        self.counter = 0
        self.node_pairs = node_pairs

        self.pair_probas = np.zeros(shape=(self.mcmc_itr-1, len(node_pairs)), dtype = np.float64)
        gt.mcmc_equilibrate(self.state, force_niter=self.mcmc_itr, mcmc_args=dict(niter=self.sweeps_per_itr),
                            callback=self.collect_edge_probas)

        scores = spsp.logsumexp(self.pair_probas, axis=0) - np.log(self.mcmc_itr - 1)
        if self.write_preds:
            self.save_file_path+=f"SBM_scores_t_{trial_num}.txt"
            with open(self.save_file_path, 'w') as f:
                f.write('drug\tdisease\tscore\tedge\n')
                for pair, score, label in zip(node_pairs, scores, labels):
                    u, v = pair
                    formatted_score = "{:.5f}".format(score)
                    f.write(f'{v}\t{u}\t{formatted_score}\t{label}\n')
            print(f'Microcanonical SBM predictions saved to: {self.save_file_path}')
            return None
        return scores
