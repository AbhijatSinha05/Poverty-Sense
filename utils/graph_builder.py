import torch
import osmnx as ox
from torch_geometric.data import Data

ox.settings.use_cache = True
ox.settings.log_console = False

def build_graph(lat, lon):
    try:
        G = ox.graph_from_point((lat, lon), dist=800, network_type="walk")
        G = ox.project_graph(G)

        node_map = {n: i for i, n in enumerate(G.nodes())}
        edges = [[node_map[u], node_map[v]] for u, v in G.edges()]

        if len(edges) == 0:
            raise ValueError("Empty graph")

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Placeholder node features (real features assigned later)
        x = torch.zeros((len(G.nodes()), 1))

        return Data(x=x, edge_index=edge_index)

    except:
        # Safe fallback
        x = torch.zeros((1,1))
        edge_index = torch.zeros((2,1), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
