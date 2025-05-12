from bipartite_link_prediction.base import BaseLinkPredictor
from bipartite_link_prediction.bipartite_network import BipartiteNetwork
import graph_tool.all as gt
import numpy as np
from torch_geometric.nn import Node2Vec
import torch
from torch_geometric.data import Data
from typing import Any, Optional, List, Tuple
import time



class N2VPredictor(BaseLinkPredictor):

    def __init__(self,
                 emb_dim: int = 128,
                 walk_length: int = 20,
                 context_size: int = 10,
                 walks_per_node: int = 10,
                 batch_size: int = 128,
                 num_epochs: int = 50,
                 lr: float = 0.01,
                 p: float = 1,
                 q: float = 1,
                 write_preds: bool = False,
                 save_file_path: Optional[str] = "./bipartite_link_prediction/data/results/") -> None:
        """
        Initialize the model with several configuration parameters.

        Args:
        emb_dim (int): Size of the embedding dimension.
        walk_length (int): Length of each walk.
        context_size (int): Size of the window of context when generating samples.
        walks_per_node (int): Number of walks to perform per node.
        batch_size (int): Size of batches during training.
        num_epochs (int): Number of epochs for which to train the model.
        lr (float): Learning rate for the optimizer.
        p (float): Return parameter for node2vec.
        q (float): In-out parameter for node2vec.
        write_preds (bool): Flag indicating whether to write predictions to a file.
        save_file_path (Optional[str]): Optional file path where predictions are to be saved.

        Returns:
        None.
        """

        self._is_data_prepared = False
        self.emb_dim = emb_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.p = p
        self.q = q
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.write_preds = write_preds
        self.save_file_path = save_file_path

    def prepare_data(self, bipartite_network: BipartiteNetwork,
                     test_edges: Optional[List[Tuple[int, int]]] = None) -> Any:
        """
        Prepares the data for training the model.
        Args:
        bipartite_network (BipartiteNetwork): The bipartite network object.
        test_edges (Optional[List[Tuple[int, int]]]): A list of test edges to exclude from training.

        Returns:
        Any: The PyG Data object containing the training data.
        """
        test_edges_set = list(map(set, test_edges))
        train_graphview = gt.GraphView(bipartite_network.graph,
                                       efilt=lambda e: set(e) not in test_edges_set)
        train_graph = gt.Graph(train_graphview, prune=True)

        train_edges = [(u, v) if train_graph.vp.node_type[u] == "disease" else (v, u) for u, v in train_graph.iter_edges()]
        train_edge_index_tensor = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
        pyg_train_data = Data(edge_index=train_edge_index_tensor, num_nodes=train_graph.num_vertices())

        self._is_data_prepared = True
        return pyg_train_data

    def _train(self) -> float:

        """
        Trains the model on the training data.
        Returns:
        float: The average loss over the training data.
        """

        self.model.train()
        total_loss = 0;
        for pos_rw, neg_rw in self.train_loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss/len(self.train_loader)

    def fit(self, data: Any) -> None:
        """
        Fits the model to the training data.
        Args:
        data (Any): The training data.

        Returns:
        None.
        """
        if not self._is_data_prepared:
            raise RuntimeError("prepare_data must be called before fit.")
        self.graph = data
        self.model = Node2Vec(self.graph.edge_index, num_nodes=self.graph.num_nodes, embedding_dim=self.emb_dim,
                              walk_length=self.walk_length, context_size=self.context_size,
                              walks_per_node=self.walks_per_node, p=self.p, q=self.q).to(self.device)
        self.train_loader = self.model.loader(batch_size=self.batch_size)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)
        for epoch in range(self.num_epochs):
            loss = self._train()
            print(f'Epoch: {epoch}, Loss: {loss:.4f}')

        self._is_data_prepared = False

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
        self.model.eval()
        with torch.no_grad():
            emb = self.model()
        emb = emb.float()
        score_matrix = torch.matmul(emb, emb.t())

        if self.write_preds:
            self.save_file_path += f"n2v_scores_t_{trial_num}.txt"
            with open(self.save_file_path, 'w') as f:
                f.write('drug\tdisease\tscore\tedge\n')
                for index, (u, v) in enumerate(node_pairs):
                    score = score_matrix[u, v].item()
                    formatted_score = "{:.5f}".format(score)
                    label = labels[index]
                    f.write(f"{v}\t{u}\t{formatted_score}\t{label}\n")
            print(f"Node2vec predictions saved to: {self.save_file_path}")
            return None
        else:
            scores = np.array([score_matrix[u, v].item() for u, v in node_pairs])

        return scores
