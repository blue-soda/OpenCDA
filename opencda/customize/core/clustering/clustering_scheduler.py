from opencda.customize.core.v2x.scheduler import Scheduler
import networkx as nx
from typing import Dict, List, Tuple, Optional
import math
from opencda.customize.core.v2x.network_manager import NetworkManager, ResourceConflictError
import opencda.customize.core.v2x.utils as utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

class ClusterBasedScheduler(Scheduler):
    """
    Scheduler based on cluster-based weighted conflict graph coloring.
    """
    def __init__(self, network_manager: 'NetworkManager'):
        super().__init__(network_manager)

        # Interference model parameters
        self.d_th = 35  # Distance threshold for interference (meters)
        self.I_max = 1.0  # Maximum possible interference contribution
        self.eta = 2.0  # Path loss exponent (environment-specific)
        self.conflict_distance = 35 # meters

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
        self.build_weighted_conflict_graph(cluster_state)

        # Step 2: Build conflict graph between communication pairs
        self.build_link_conflict_graph()

        # Step 3: Perform graph coloring to assign subchannels
        self.assign_subchannels()

        self.draw_weighted_conflict_graph()

    def build_weighted_conflict_graph(self, cluster_state: Dict):
        """
        Build a directed graph where a directed edge (i → j) indicates
        that vehicle i will cause interference to vehicle j if it transmits.

        The edge weight quantifies how severe the interference is and may
        be used to prioritize certain links or assess impact.

        Args:
            cluster_state (Dict): Contains 'members' and 'neighbors' (vehicles)
        """
        self.weighted_conflict_graph = nx.DiGraph()
        members = cluster_state['members']
        neighbors = cluster_state['neighbors']

        # Add all vehicle nodes to the graph
        for vid in set(list(members.keys()) + list(neighbors.keys())):
            self.weighted_conflict_graph.add_node(vid)

        # Handle interference between member vehicles
        for i_id, i in members.items():
            for j_id, j in members.items():
                if i_id == j_id:
                    continue
                if self.is_conflicting(i, j):
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

    def build_link_conflict_graph(self):
        """
        Build a conflict graph where:
        - Each node represents a communication link (i → j)
        - Two nodes are connected if their transmissions interfere with each other
        """
        self.conflict_graph.clear()
        self.communication_edges.clear()
        all_edges = list(self.weighted_conflict_graph.edges())

        # Each directed communication (i → j) forms a communication node
        for (u, v) in all_edges:
            if u == v:
                continue
            self.conflict_graph.add_node((u, v))
            self.communication_edges.add((u, v))

        edges_list = list(self.communication_edges)
        n = len(edges_list)

        # Check pairwise conflicts between communication edges
        for i in range(n):
            u1, v1 = edges_list[i]
            for j in range(i + 1, n):
                u2, v2 = edges_list[j]
                conflict = False

                # Rule 1: Same transmitter (cannot use same subchannel)
                if u1 == u2:
                    conflict = True
                # Rule 2: u1's transmission interferes with v2's reception or vice versa
                elif self.weighted_conflict_graph.has_edge(u1, v2):
                    conflict = True
                elif self.weighted_conflict_graph.has_edge(u2, v1):
                    conflict = True

                if conflict:
                    self.conflict_graph.add_edge((u1, v1), (u2, v2))

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
            return -1, -1, -1, False  # communication edge not activated
        # Step 2: Attempt to allocate resource using the coloring result (subchannel)
        subchannel = self.coloring[key]
        try:
            # Call the underlying resource manager to allocate time slot and subchannel
            subchannel, start_slot, end_slot = self.network_manager.allocate_resource(
                source, target, volume, subchannel
            )
            return subchannel, start_slot, end_slot, True
        except ResourceConflictError as e:
            print(f"[ClusterBasedScheduler] Resource conflict: {e}")
            return -1, -1, -1, False


    def draw_weighted_conflict_graph(self):
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

        filename='link_conflict_graph.png'
        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(self.conflict_graph, seed=42)

        nx.draw_networkx_nodes(self.conflict_graph, pos, node_size=600, node_color='lightgreen')
        nx.draw_networkx_labels(self.conflict_graph, pos, font_size=8)
        edge_labels = nx.get_edge_attributes(self.conflict_graph, 'weight')
        nx.draw_networkx_edges(self.conflict_graph, pos, edge_color='black')
        nx.draw_networkx_edge_labels(self.conflict_graph, pos, edge_labels={k: f'{v:.2f}' for k, v in edge_labels.items()}, font_size=7)
        plt.title('Conflict Graph (Link-level with Weights)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        filename='coloring_graph.png'
        plt.figure(figsize=(12, 10))
        G = nx.Graph()
        # 1. Build nodes based on coloring
        for link, color in self.coloring.items():
            G.add_node(link, color=color)
        # 2. Optional: Add conflicts as edges (from conflict graph)
        if hasattr(self, 'conflict_graph'):
            for u, v in self.conflict_graph.edges():
                if u in G.nodes and v in G.nodes:
                    G.add_edge(u, v)
        # 3. Extract colors
        color_values = [G.nodes[n]['color'] for n in G.nodes()]
        num_colors = max(color_values) + 1 if color_values else 1
        cmap = cm.get_cmap('tab20', num_colors)  # 20 个颜色支持
        norm = mcolors.Normalize(vmin=0, vmax=num_colors - 1)
        node_colors = [cmap(norm(c)) for c in color_values]
        # 4. plot
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos,
                            node_color=node_colors,
                            node_size=800)
        nx.draw_networkx_labels(G, pos,
                                labels={n: f"{n[0]}→{n[1]}" for n in G.nodes()},
                                font_size=8)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=range(num_colors))
        cbar.set_label("Subchannel (Color Index)")
        cbar.ax.set_yticklabels([str(i) for i in range(num_colors)])
        plt.title("Communication Link Coloring Result")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()