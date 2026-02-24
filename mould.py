"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from queue import PriorityQueue

# -----------------------
# Parameters
# -----------------------
N_AGENTS = 5
EVAPORATION_RATE = 0.1
MIN_CONCENTRATION = 1e-6

# -----------------------
# Initialize graph
# -----------------------
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (4, 5), (3, 7), (7, 6), (5, 9), (0, 8), (8, 2), (6, 3), (6, 4), (5, 7)])

n_nodes = len(G.nodes)

# -----------------------
# Initialize state
# -----------------------
plasmoid = np.zeros((N_AGENTS, n_nodes))
slime_concentration = np.zeros((N_AGENTS, n_nodes))

start_nodes = np.random.choice(list(G.nodes), size=N_AGENTS)
end_node = np.random.choice(list(G.nodes))

for i, s in enumerate(start_nodes):
    plasmoid[i, s] = 1.0

# Store animation frames
frames = []

# -----------------------
# Adaptive Edge Memory
# -----------------------
edge_pheromone = {tuple(sorted(edge)): 1.0 for edge in G.edges()}

PRUNE_THRESHOLD = 0.05
REINFORCE_RATE = 0.3
MAX_STEPS = 30


def adaptive_forage(start_node):
    active_nodes = {start_node}
    plasmoid = np.zeros(n_nodes)
    plasmoid[start_node] = 1.0

    for step in range(MAX_STEPS):
        new_active = set()
        delta_pheromone = {}

        for node in active_nodes:
            neighbors = list(G.neighbors(node))

            # Only consider discovered edges
            valid_neighbors = []
            weights = []

            for n in neighbors:
                edge = tuple(sorted((node, n)))
                if edge in edge_pheromone:
                    valid_neighbors.append(n)
                    weights.append(edge_pheromone[edge])

            if not valid_neighbors:
                continue

            weights = np.array(weights)
            weights = weights / weights.sum()

            for neighbor, w in zip(valid_neighbors, weights):
                flow = plasmoid[node] * w

                if flow > MIN_CONCENTRATION:
                    plasmoid[neighbor] += flow
                    new_active.add(neighbor)

                    edge = tuple(sorted((node, neighbor)))
                    delta_pheromone[edge] = delta_pheromone.get(edge, 0) + flow

        # --- Reinforce successful edges ---
        for edge, reinforcement in delta_pheromone.items():
            edge_pheromone[edge] += REINFORCE_RATE * reinforcement

        # --- Evaporate pheromone ---
        for edge in edge_pheromone:
            edge_pheromone[edge] *= (1 - EVAPORATION_RATE)

        # --- Prune weak edges ---
        to_remove = [e for e, v in edge_pheromone.items() if v < PRUNE_THRESHOLD]
        for e in to_remove:
            del edge_pheromone[e]

        active_nodes = new_active

        frames.append(plasmoid.copy())

        if end_node in active_nodes:
            break

    return plasmoid

def find_strongest_path(start, end):
    path = [start]
    visited = {start}

    while path[-1] != end:
        current = path[-1]

        neighbors = list(G.neighbors(current))
        best_neighbor = None
        best_weight = 0

        for n in neighbors:
            if n in visited:
                continue

            edge = tuple(sorted((current, n)))
            weight = edge_pheromone.get(edge, 0)

            if weight > best_weight:
                best_weight = weight
                best_neighbor = n

        if best_neighbor is None:
            break

        path.append(best_neighbor)
        visited.add(best_neighbor)

    return path

# -----------------------
# Run propagation
# -----------------------
for start_node in start_nodes:
    adaptive_forage(start_node)

# -----------------------
# Find optimal path
# -----------------------
def find_path(start_node, end_node):
    total_concentration = np.sum(plasmoid, axis=0)
    total_concentration *= (1 - EVAPORATION_RATE)

    path = [start_node]
    visited = {start_node}

    while path[-1] != end_node:
        current = path[-1]
        neighbors = list(G.neighbors(current))

        unvisited = [n for n in neighbors if n not in visited]
        if not unvisited:
            break

        concentrations = [total_concentration[n] for n in unvisited]
        next_node = unvisited[np.argmax(concentrations)]

        path.append(next_node)
        visited.add(next_node)

    return path

optimal_path = find_path(start_nodes[0], end_node)

# -----------------------
# Animation
# -----------------------
pos = nx.spring_layout(G)
fig, ax = plt.subplots()

def update(frame):
    ax.clear()

    total_conc = frame

    # --- Node colors ---
    node_colors = total_conc

    # --- Edge intensities ---
    edge_values = []
    for u, v in G.edges():
        # Use average slime at connected nodes
        edge_strength = (total_conc[u] + total_conc[v]) / 2
        edge_values.append(edge_strength)

    # Normalize edge values
    max_val = max(edge_values) if max(edge_values) > 0 else 1
    edge_colors = [val / max_val for val in edge_values]
    edge_widths = [1 + 5 * (val / max_val) for val in edge_values]

    # Draw graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=800,
        edge_color=edge_colors,
        edge_cmap=plt.cm.plasma,
        width=edge_widths,
        ax=ax
    )

    ax.set_title("Slime Mold Propagation (Edge Flow Visualization)")

ani = FuncAnimation(fig, update, frames=frames, interval=700, repeat=False)
plt.show()

print("Start nodes:", start_nodes)
print("End node:", end_node)
print("Optimal path:", optimal_path)
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# Parameters
# -----------------------
N_AGENTS = 5
EVAPORATION_RATE = 0.1
REINFORCE_RATE = 0.3
PRUNE_THRESHOLD = 0.05
MIN_CONCENTRATION = 1e-6
MAX_STEPS = 30

EVAPORATION_RATE = 0.03     # was 0.1 (too high)
REINFORCE_RATE = 0.6        # stronger reinforcement
PRUNE_THRESHOLD = 0.01      # prune less aggressively
MAX_STEPS = 60              # allow convergence time

# -----------------------
# Initialize graph
# -----------------------
G = nx.Graph()
G.add_edges_from([
    # Main backbone corridor
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Parallel alternative path
    (0, 5), (5, 6), (6, 3),

    # Long detour loop
    (1, 7), (7, 8), (8, 9), (9, 4),

    # Thin shortcut (should compete strongly)
    (1, 10), (10, 4),

    # Cross-connectors to create competition
    (5, 2),
    (7, 6),
    (8, 3),

    # Dead-end branches (should prune)
    (2, 11),
    (6, 12),
    (8, 13),

    # Previously isolated region — now connected weakly
    (4, 14),
    (14, 15),
    (15, 9)
])

n_nodes = len(G.nodes)

# -----------------------
# Initialize pheromone memory
# -----------------------
edge_pheromone = {tuple(sorted(edge)): 1.0 for edge in G.edges()}

# -----------------------
# Initialize start/end
# -----------------------
start_nodes = np.random.choice(list(G.nodes), size=N_AGENTS)
end_node = np.random.choice(list(G.nodes))

# -----------------------
# Animation storage
# -----------------------
frames = []

# -----------------------
# Adaptive Foraging Function
# -----------------------
def adaptive_forage(start_node):
    active_nodes = {start_node}
    local_plasmoid = np.zeros(n_nodes)
    local_plasmoid[start_node] = 1.0

    for step in range(MAX_STEPS):
        new_active = set()
        delta_pheromone = {}

        for node in active_nodes:
            neighbors = list(G.neighbors(node))

            valid_neighbors = []
            weights = []

            for n in neighbors:
                edge = tuple(sorted((node, n)))
                if edge in edge_pheromone:
                    valid_neighbors.append(n)
                    weights.append(edge_pheromone[edge])

            if not valid_neighbors:
                continue

            weights = np.array(weights)
            weights = weights / weights.sum()

            for neighbor, w in zip(valid_neighbors, weights):
                flow = local_plasmoid[node] * w

                if flow > MIN_CONCENTRATION:
                    local_plasmoid[neighbor] += flow
                    new_active.add(neighbor)

                    edge = tuple(sorted((node, neighbor)))
                    delta_pheromone[edge] = delta_pheromone.get(edge, 0) + flow

        # Reinforce successful edges
        for edge, reinforcement in delta_pheromone.items():
            edge_pheromone[edge] += REINFORCE_RATE * reinforcement

        # Evaporate pheromone
        for edge in list(edge_pheromone.keys()):
            edge_pheromone[edge] *= (1 - EVAPORATION_RATE)

        # Prune weak edges
        for edge in list(edge_pheromone.keys()):
            if edge_pheromone[edge] < PRUNE_THRESHOLD:
                del edge_pheromone[edge]

        active_nodes = new_active

        frames.append(local_plasmoid.copy())

        if end_node in active_nodes:
            break

# -----------------------
# Run Foraging Agents
# -----------------------
for start_node in start_nodes:
    adaptive_forage(start_node)

# -----------------------
# Extract Strongest Path
# -----------------------
def find_strongest_path(start, end):
    path = [start]
    visited = {start}

    while path[-1] != end:
        current = path[-1]
        neighbors = list(G.neighbors(current))

        best_neighbor = None
        best_weight = 0

        for n in neighbors:
            if n in visited:
                continue

            edge = tuple(sorted((current, n)))
            weight = edge_pheromone.get(edge, 0)

            if weight > best_weight:
                best_weight = weight
                best_neighbor = n

        if best_neighbor is None:
            break

        path.append(best_neighbor)
        visited.add(best_neighbor)

    return path

optimal_path = find_strongest_path(start_nodes[0], end_node)

# -----------------------
# Visualization
# -----------------------
pos = nx.spring_layout(G)
fig, ax = plt.subplots()

def update(frame):
    ax.clear()

    node_colors = frame

    # Only draw surviving edges
    surviving_edges = list(edge_pheromone.keys())

    edge_values = []
    for u, v in surviving_edges:
        edge_values.append(edge_pheromone[(u, v)])

    if edge_values:
        max_val = max(edge_values)
        edge_colors = [v / max_val for v in edge_values]
        edge_widths = [1 + 5 * (v / max_val) for v in edge_values]
    else:
        edge_colors = []
        edge_widths = []

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=800,
        ax=ax
    )

    nx.draw_networkx_labels(G, pos, ax=ax)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=surviving_edges,
        edge_color=edge_colors,
        edge_cmap=plt.cm.plasma,
        width=edge_widths,
        ax=ax
    )

    ax.set_title("Adaptive Slime Mold Foraging with Pruning")

ani = FuncAnimation(fig, update, frames=frames, interval=700, repeat=False)
plt.show()

# -----------------------
# Output Results
# -----------------------
print("Start nodes:", start_nodes)
print("End node:", end_node)
print("Strongest path found:", optimal_path)