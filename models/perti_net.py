# perti_net.py
# Petri net model for the VidSim system, visualizing video pipeline/transmission stages as graph nodes and transitions.
# Helps understanding discrete event flow from capture through video reconstruction, and supports token-based state analysis.

import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph structure for Petri net
G = nx.DiGraph()

# Define place and transition nodes for modeling each simulation stage
places = [
    "Frame Captured", "Frame Encoded", "Packets Created", "Packets Ready to Send", "Packets Sent",
    "Packets Received", "Frame Decoded", "Video Reconstructed"
]
transitions = [
    "Capture Frame", "Encode Frame", "Packetize Frame", "Send Packets", "Receive Packets", "Decode Frame", "Reconstruct Video"
]

# Add nodes for places and transitions, using custom attribute for type
for place in places:
    G.add_node(place, type='place')
for transition in transitions:
    G.add_node(transition, type='transition')

# Arcs define allowable state transitions in pipeline
arcs = [
    ("Frame Captured", "Capture Frame"),
    ("Capture Frame", "Frame Encoded"),
    ("Frame Encoded", "Encode Frame"),
    ("Encode Frame", "Packets Created"),
    ("Packets Created", "Packetize Frame"),
    ("Packetize Frame", "Packets Ready to Send"),
    ("Packets Ready to Send", "Send Packets"),
    ("Send Packets", "Packets Sent"),
    ("Packets Sent", "Receive Packets"),
    ("Receive Packets", "Packets Received"),
    ("Packets Received", "Decode Frame"),
    ("Decode Frame", "Frame Decoded"),
    ("Frame Decoded", "Reconstruct Video"),
    ("Reconstruct Video", "Video Reconstructed")
]

# Add arcs/edges to the graph
for arc in arcs:
    G.add_edge(*arc)

# Visualize and label the Petri net/calculate initial token distribution

def visualize_petri_net(G, token_places=None):
    """
    Visualize Petri net graph, showing places and transitions with node shape and current tokens.
    Args:
        G: NetworkX DiGraph with appropriate node attributes
        token_places: Optional dict for overlaying token counts in places.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    # Draw place and transition nodes distinctly
    place_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'place']
    transition_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'transition']
    nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, node_shape='o', node_color='lightblue', node_size=500, label='Places')
    nx.draw_networkx_nodes(G, pos, nodelist=transition_nodes, node_shape='s', node_color='lightgreen', node_size=500, label='Transitions')
    nx.draw_networkx_edges(G, pos)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    # Overlay tokens if provided
    if token_places:
        for place in token_places:
            x, y = pos[place]
            plt.text(x, y+0.1, f'Tokens: {token_places[place]}', fontsize=12, ha='center', color='red')
    plt.legend(scatterpoints=1)
    plt.title('Petri Net Model of the SenseVid System')
    plt.show()

# Initial state: tokens for pending places, can be modified for simulation
# Usage: adjust token_places for different pipeline start/end states in research.
token_places = {
    "Frame Captured": 1,
    "Packets Sent": 1
}

if __name__ == "__main__":
    # Run visualize if executed as main script
    visualize_petri_net(G, token_places)
