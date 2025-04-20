from opencda.customize.core.v2x.scheduler import Scheduler
import networkx as nx
from typing import Dict, List, Tuple, Optional
import math
from opencda.customize.core.v2x.network_manager import NetworkManager, ResourceConflictError
import opencda.customize.core.v2x.utils as utils

class ClusterBasedScheduler(Scheduler):
    """
    Scheduler based on cluster-based weighted conflict graph coloring.
    """
    def __init__(self, network_manager: 'NetworkManager'):
        super().__init__(network_manager)
        self.d_th = 35  # Interference critical distance (meters)
        self.I_max = 1.0  # Maximum interference strength
        self.eta = 2.0  # Path loss exponent
        # Member variables to save graph and coloring results
        self.cluster_state = {}
        self.weighted_conflict_graph = nx.Graph()  # Weighted conflict graph
        self.coloring = {}  # Graph coloring results

    def update_scheduler(self, cluster_state: Dict):
        """
        Update the scheduler's internal state based on the provided cluster_state.
        Args:
            cluster_state (Dict): The cluster_state dictionary from a V2XManager.
        """
        self.cluster_state = cluster_state
        # Step 1: Build the weighted conflict graph
        self.build_weighted_conflict_graph(cluster_state)
        # Step 2: Perform weighted graph coloring
        self.weighted_graph_coloring()

    def build_weighted_conflict_graph(self, cluster_state: Dict):
        """
        Build the weighted conflict graph based on the provided cluster_state.
        Args:
            cluster_state (Dict): The cluster_state dictionary.
        """
        self.weighted_conflict_graph = nx.Graph()
        cluster_head = cluster_state['cluster_head']
        members = cluster_state['members']
        neighbors = cluster_state['neighbors']
        # Add nodes and intra-cluster conflict edges
        
        for member_id, member in members.items():
            for k in range(self.network_manager.subchannels): 
                node = (member_id, k)
                self.weighted_conflict_graph.add_node(node, weight=cluster_state['priority_score'])
                # Add intra-cluster conflict edges
                for other_member_id, other_member in members.items():
                    if other_member_id != member_id and self.is_conflicting(member, other_member, k):
                        d_ij = utils.calculate_distance(member, other_member)
                        I_ij_k = utils.get_interference_contribution(member, other_member, k)
                        edge_weight = (d_ij / self.d_th) * (I_ij_k / self.I_max)
                        self.weighted_conflict_graph.add_edge((member_id, k), (other_member_id, k), weight=edge_weight)

        # Add nodes and inter-cluster conflict edges
        for neighbor_id, neighbor in neighbors.items():
            for k in range(self.network_manager.subchannels):
                node = (neighbor_id, k)
                self.weighted_conflict_graph.add_node(node, weight=neighbor['priority_score'])
                # Add inter-cluster conflict edges
                for member_id, member in members.items():
                    d_ij = utils.calculate_distance(member, neighbor)
                    edge_weight = (d_ij ** self.eta) / (self.d_th ** self.eta)
                    self.weighted_conflict_graph.add_edge((member_id, k), (neighbor_id, k), weight=edge_weight)


    def weighted_graph_coloring(self):
        """
        Perform weighted graph coloring on the current weighted_conflict_graph.
        """
        self.coloring = nx.greedy_color(self.weighted_conflict_graph, strategy='largest_first',
                                       weight=lambda e: self.weighted_conflict_graph.edges[e]['weight'])
        

    def schedule(self, source, target, volume: float) -> Tuple[int, int, int, bool]:
        """
        Schedule resources using cluster-based weighted conflict graph coloring.
        Args:
            source (V2XManager): The source vehicle manager.
            target (V2XManager): The target vehicle manager.
            volume (float): The data volume to transmit (in MB).
        Returns:
            Tuple[int, int, int, bool]: A tuple containing:
                - subchannel: The allocated subchannel index.
                - start_time_slot: The starting time slot for the communication.
                - end_time_slot: The ending time slot for the communication.
                - success: Whether the allocation was successful.
        """
        nm = self.network_manager
        if nm is None:
            return -1, -1, -1, False  # NetworkManager has been garbage collected
        # Step 1: Allocate resources based on coloring
        subchannel = self.coloring.get((source.id, 0), -1)  # Example: Use the first RBG
        if subchannel == -1:
            return -1, -1, -1, False
        try:
            subchannel, start_time_slot, end_time_slot = nm.allocate_resource(source, target, volume, subchannel)
            return subchannel, start_time_slot, end_time_slot, True
        except ResourceConflictError as e:
            print(f"ClusterBasedScheduler: {e}")
            return -1, -1, -1, False
        

    def is_conflicting(self, i, j, k: int) -> bool:
        """
        Check if two vehicles conflict on a subchannel.
        i: 'V2XManager', j: 'V2XManager'
        """
        # Placeholder for conflict detection
        return self.get_distance(i, j) <= self.d_th