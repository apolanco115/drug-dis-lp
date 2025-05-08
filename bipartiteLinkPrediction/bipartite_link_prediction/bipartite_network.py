from collections import defaultdict
from typing import Optional, Dict, List

import graph_tool.all as gt
import numpy as np


class BipartiteNetwork:
    def __init__(self,
                 graph: Optional[gt.Graph] = None,
                 file_path: Optional[str] = None,
                 node_type_map: Optional[Dict[str, int]] = None):
        """
        Initializes a BipartiteNetwork instance either by loading a graph from a file or by using
        an existing graph-tool graph object. If both `graph` and `file_path` are provided, `graph`
        takes precedence.

        Args:
        graph (Optional[gt.Graph]): A graph-tool Graph instance to use for the bipartite network.
        file_path (Optional[str]): Path to the file from which to load the graph.
        node_type_map (Optional[Dict[str, int]]): A mapping of node types to 0 or 1. If not provided,

        Returns:
        None
        """
        if graph is not None:
            self.graph = self._validate_graph(graph)
        elif file_path is not None:
            self.graph = self._load_graph_from_file(file_path)
        else:
            raise ValueError("Either 'graph' or 'file_path' must be provided to initialize the BipartiteNetwork.")

        # Verify if the graph is indeed bipartite
        if not gt.is_bipartite(self.graph):
            raise ValueError("The provided graph is not bipartite.")

        self.top_nodes = []
        self.bottom_nodes = []
        self.node_type = np.zeros(self.graph.num_vertices(), dtype=np.int32)
        self.node_type_map = node_type_map
        self.all_edges = []
        self._partition_nodes()
        self._set_all_edges()
        self._create_adjacency_lists()


    def _validate_graph(self, graph: gt.Graph) -> gt.Graph:
        """
        Validates the provided graph-tool Graph instance.
        Args:
        graph (gt.Graph): A graph-tool Graph instance to validate.

        Returns:
        gt.Graph: The validated graph-tool Graph instance.
        """
        if not isinstance(graph, gt.Graph):
            raise ValueError("The 'graph' argument must be an instance of 'graph_tool.Graph'.")
        return graph

    def _load_graph_from_file(self, file_path: str) -> gt.Graph:
        """
        Loads a graph-tool Graph from a file.

        Args:
        file_path (str): Path to the file from which to load the graph.

        Returns:
        gt.Graph: The loaded graph-tool Graph instance.
        """
        try:
            return gt.load_graph(file_path)
        except IOError as e:
            raise IOError(f"Could not load the graph from the specified path: {file_path}") from e

    def _partition_nodes(self) -> None:
        """
        Partitions the graph into top_nodes and bottom_nodes lists based on the 'node_type' vertex property.
        If no mapping is provided, partitions according to the 'node_type' property with values 0 or 1.
        If a mapping is provided, it is used to translate 'node_type' values to 0 or 1.

        Returns:
        None
        """
        # Use graph_tool.is_bipartite to create a partition if no node_type_map is provided
        if not self.node_type_map:
            _, node_type_prop = gt.is_bipartite(self.graph, partition=True)
        else:
            if 'node_type' not in self.graph.vertex_properties:
                raise ValueError("Graph does not have 'node_type' vertex property.")
            node_type_prop = self.graph.vertex_properties['node_type']

        # Assign vertices to top_nodes or bottom_nodes based on node types or provided map
        for v in self.graph.iter_vertices():
            # map the node type to 0/1 if node_type_map is provided, else use existing values
            if self.node_type_map:
                node_type_value = self.node_type_map[node_type_prop[v]]
                if node_type_value not in [0, 1]:
                    raise ValueError(f"Unrecognized node_type value: {node_type_prop[v]}. Values must either 0 or 1.")
            else:
                node_type_value = node_type_prop[v]

            if node_type_value == 0:
                self.top_nodes.append(int(v))
                self.node_type[int(v)] = 0
            elif node_type_value == 1:
                self.bottom_nodes.append(int(v))
                self.node_type[int(v)] = 1
                
    def _create_adjacency_lists(self) -> None:
        """
        Creates adjacency lists for all nodes. Each node's list contains its direct neighbors.

        """
        self.adjacency_list_top = defaultdict(list)  # Adjacency list for top nodes
        self.adjacency_list_bottom = defaultdict(list)  # Adjacency list for bottom nodes

        # Create adjacency list for top nodes
        for v in self.top_nodes:
            v_neighbors = self.graph.get_out_neighbors(v)
            self.adjacency_list_top[v].extend(v_neighbors)

        # Create adjacency list for bottom nodes
        for v in self.bottom_nodes:
            v_neighbors = self.graph.get_out_neighbors(v)
            self.adjacency_list_bottom[v].extend(v_neighbors)

    def _set_all_edges(self) -> None:
        """
        Sets the all_edges attribute to a list of all edges in the bipartite network.

        Returns:
        None
        """
        # all edges should be of the form: (top_nodes, bottom_nodes)
        for edge in self.graph.iter_edges():
            n1, n2 = edge
            if self.node_type[int(n1)] == 0:
                self.all_edges.append((int(n1), int(n2)))
            else:
                self.all_edges.append((int(n2), int(n1)))