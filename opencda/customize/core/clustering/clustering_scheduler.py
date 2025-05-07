from opencda.customize.core.v2x.scheduler import Scheduler
import networkx as nx
from typing import Dict, List, Tuple, Optional
import math
from opencda.customize.core.v2x.network_manager import NetworkManager, ResourceConflictError
import opencda.customize.core.v2x.utils as utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from opencda.log.logger_config import logger
from random import uniform

class ClusterBasedScheduler(Scheduler):
    """
    Scheduler based on cluster-based weighted conflict graph coloring.
    """
    def __init__(self, network_manager: 'NetworkManager', config={}):
        super().__init__(network_manager, config)

        # Interference model parameters
        self.I_max = 1.0  # Maximum possible interference contribution
        self.eta = 2.0  # Path loss exponent (environment-specific)

        self.conflict_distance = config.get('conflict_distance', 70) # meters
        self.d_th = self.conflict_distance  # Distance threshold for interference (meters)
        self.M = config.get('M', 5)
        self.subchannel_num = config.get('subchannel_num', 20)
        self.offset = int(uniform(0.0, 1.0) * self.subchannel_num)

        # Internal scheduler state
        self.cluster_state = {}  # Stores members and neighbors from the cluster
        self.weighted_conflict_graph = nx.DiGraph()  # Vehicle-level interference graph
        self.conflict_graph = nx.Graph()  # Communication-pair-level conflict graph
        self.communication_edges = set()  # Set of valid communication links (i → j)
        self.coloring = {}  # Mapping of communication edges → subchannel (color)

    def update_scheduler(self, cluster_state: Dict):
        """
        Main interface: update the scheduler with current cluster state
        and perform subchannel assignment.

        Args:
            cluster_state (Dict): Cluster info (members, neighbors, positions)
        """
        self.cluster_state = cluster_state

        # Step 1: Build vehicle-level interference graph based on wireless model
        self.build_weighted_conflict_graph()

        # Step 2: Build conflict graph between communication pairs
        self.weighted_coloring_upload_scheduling()

        # Step 3: Perform graph coloring to assign subchannels
        # self.assign_subchannels()

        # self.draw_weighted_conflict_graph()

    def build_weighted_conflict_graph(self):
        """
        Build a directed graph where a directed edge (i → j) indicates
        that vehicle i will cause interference to vehicle j if it transmits.

        The edge weight quantifies how severe the interference is and may
        be used to prioritize certain links or assess impact.

        Args:
            cluster_state (Dict): Contains 'members' and 'neighbors' (vehicles)
        """
        self.weighted_conflict_graph = nx.DiGraph()
        members = self.cluster_state['members']
        neighbors = self.cluster_state['neighbors']
        self.cluster_head = self.cluster_state['cluster_head'].vehicle_id

        # Add all vehicle nodes to the graph
        for vid in set(list(members.keys()) + list(neighbors.keys())):
            self.weighted_conflict_graph.add_node(vid)

        # Handle interference between member vehicles
        for i_id, i in members.items():
            for j_id, j in members.items():
                if i_id == j_id:
                    continue
                # if self.is_conflicting(i, j):
                d_ij = utils.calculate_distance(i, j)
                I_ij_k = utils.get_interference_contribution(i, j)
                edge_weight = (d_ij / self.d_th) * (I_ij_k / self.I_max)
                self.weighted_conflict_graph.add_edge(i_id, j_id, weight=edge_weight)

        # Handle interference from members to external neighbors
        for i_id, i in members.items():
            for j_id, j in neighbors.items():
                if j_id in members:
                    continue
                if self.is_conflicting(i, j):
                    d_ij = utils.calculate_distance(i, j)
                    edge_weight = (d_ij ** self.eta) / (self.d_th ** self.eta)
                    self.weighted_conflict_graph.add_edge(i_id, j_id, weight=edge_weight)

    def is_conflicting(self, sender, receiver) -> bool:
        """
        Determine if transmission from sender will cause too much interference
        to the receiver, based on the path-loss model.

        Returns:
            bool: True if interference >= threshold, meaning conflict
        """
        distance = utils.calculate_distance(sender, receiver)
        return distance <= self.conflict_distance

    def weighted_coloring_upload_scheduling(self, eps=0.0):
        """
        Assign vehicles to different subchannels using graph coloring.
        Each vehicle must occupy a different subchannel.
        There are exactly M subchannels available.
        Using greedy coloring from networkx.

        Args:
            eps (float): Interference tolerance (not used directly here since strict 1-to-1 mapping).
        """

        uploads = [vid for vid in self.cluster_state['members'].keys() if vid != self.cluster_head]
        
        # Step 1: Build the effective conflict graph among uploads
        G = nx.Graph()

        G.add_nodes_from(uploads)

        for i in uploads:
            for j in uploads:
                if i == j:
                    continue
                # If interference between i and j is nonzero, add an undirected edge
                w = 0
                if self.weighted_conflict_graph.has_edge(i, j):
                    w += self.weighted_conflict_graph[i][j]['weight']
                if self.weighted_conflict_graph.has_edge(j, i):
                    w += self.weighted_conflict_graph[j][i]['weight']
                if w > eps:
                    G.add_edge(i, j, weight=w)

        # Step 2: Perform greedy coloring
        # strategy: smallest_last, largest_first, random_sequential, saturation_largest_first
        # smallest_last tends to use fewer colors
        coloring_result = nx.coloring.greedy_color(G, strategy='smallest_last')

        logger.debug(f"{self.cluster_head}'s coloring_result: {coloring_result}")
        # Step 3: Check if we used too many colors
        if len(coloring_result) == 0:
            max_color_used = 1
        else:
            max_color_used = max(coloring_result.values()) + 1  # from 0
        if max_color_used > self.M:
            logger.warning(f"Cannot schedule! Required subchannels ({max_color_used}) > available M ({self.M}).")

        # Step 4: Save formatting: (u, head) -> color
        coloring = dict()
        for u, color in coloring_result.items():
            coloring[(u, self.cluster_head)] = color

        self.coloring = coloring


    def assign_subchannels(self):
        """
        Perform greedy graph coloring on the link conflict graph,
        assigning a color (subchannel) to each communication link node.
        """
        self.coloring = nx.coloring.greedy_color(self.conflict_graph, strategy="largest_first")
        return self.coloring  # {(u, v): subchannel_id}

    def schedule(self, source, target, volume: float) -> Tuple[int, int, int, bool]:
        """
        Query if the communication from sender_id to receiver_id has been scheduled
        (i.e., a subchannel assigned).

        Returns:
            (subchannel_id: int, valid: bool): If scheduled, return assigned color;
                                            else (-1, False)
        """
        key = (source.vehicle_id, target.vehicle_id)
        # Step 1: Check if the link was already assigned a color during coloring
        if key not in self.coloring:
            logger.error(f"key {key} not in self.coloring {self.coloring}")
            return -1, -1, -1, False  # communication edge not activated
        # Step 2: Attempt to allocate resource using the coloring result (subchannel)
        subchannel = (self.coloring[key] + self.offset) % self.subchannel_num

        # try:
        # Call the underlying resource manager to allocate time slot and subchannel
        subchannel, start_slot, end_slot = self.network_manager.allocate_resource(
            source, target, volume, subchannel 
        )

        success = subchannel >= 0
        if start_slot == 100: #debug
            self.visualize_weighted_conflict_graph()
            self.visualize_coloring()

        if not success: #Reset the offset if failed
            self.offset = int(uniform(0.0, 1.0) * self.subchannel_num)

        return subchannel, start_slot, end_slot, success
        
        # except ResourceConflictError as e:
        #     print(f"[ClusterBasedScheduler] Resource conflict: {e}")
        #     return -1, -1, -1, False


    def visualize_weighted_conflict_graph(self):
        """
        Visualize and save the graph.
        """
        filename='weighted_conflict_graph.png'
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.weighted_conflict_graph, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(self.weighted_conflict_graph, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(self.weighted_conflict_graph, pos)
        # Edge weights
        edge_labels = nx.get_edge_attributes(self.weighted_conflict_graph, 'weight')
        # Draw edges with arrows
        nx.draw_networkx_edges(self.weighted_conflict_graph, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_edge_labels(self.weighted_conflict_graph, pos, edge_labels={k: f'{v:.2f}' for k, v in edge_labels.items()})
        plt.title("Weighted Conflict Graph (Vehicle-level)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def visualize_coloring(self):
        """
        Visualize the result of self.coloring on the full weighted_conflict_graph.
        Nodes without assigned color will be shown in gray.
        """
        if not hasattr(self, 'coloring'):
            print("No coloring found! Please run weighted_coloring_upload_scheduling first.")
            return

        # Use the full weighted conflict graph
        G = self.weighted_conflict_graph

        # Extract node -> color mapping
        node_colors = dict()
        for (v, head), color in self.coloring.items():
            node_colors[v] = color

        # Prepare color mapping
        unique_color_ids = sorted(set(node_colors.values()))
        cmap = plt.get_cmap('tab20')  # Colormap with 20 distinct colors
        color_map = {color_id: cmap(i / max(1, len(unique_color_ids)-1)) for i, color_id in enumerate(unique_color_ids)}

        # Generate color list for all nodes
        node_color_list = []
        for v in G.nodes():
            if v in node_colors:
                node_color_list.append(color_map[node_colors[v]])
            else:
                node_color_list.append('lightgray')  # Default color for unassigned nodes

        # Plot the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)  # Layout positions

        nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=500, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels={v: str(v) for v in G.nodes()}, font_color='black')

        plt.title("Weighted Upload Scheduling Result (Full Graph)")
        plt.axis('off')
        # plt.show()
        plt.savefig('coloring.png')
        plt.close()
