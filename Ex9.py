import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.matching import maximum_matching


def generate_balanced_matrix(n):
    mat = np.zeros((n, n))
    for i in range(n):
        remaining = 1.0
        indices = list(range(n))
        random.shuffle(indices)
        for j in indices[:-1]:
            val = round(random.uniform(0, remaining), 2)
            mat[i][j] = val
            remaining -= val
        mat[i][indices[-1]] = round(remaining, 2)
    for j in range(n):
        col_sum = mat[:, j].sum()
        mat[:, j] /= col_sum
    return pd.DataFrame(mat, columns=[f"Item {j+1}" for j in range(n)],
                        index=[f"Player {i+1}" for i in range(n)])


def is_balanced(matrix):
    return (np.allclose(matrix.sum(axis=1), 1) and
            np.allclose(matrix.sum(axis=0), 1))


def draw_graph(matrix, title):
    G = nx.Graph()
    n = len(matrix)
    for i in range(n):
        G.add_node(f"P{i+1}", bipartite=0)
    for j in range(n):
        G.add_node(f"I{j+1}", bipartite=1)
        for i in range(n):
            weight = matrix.iloc[i, j]
            if weight > 1e-8:
                G.add_edge(f"P{i+1}", f"I{j+1}", weight=round(weight, 2))
    pos = nx.bipartite_layout(G, [f"P{i+1}" for i in range(n)])
    labels = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)
    plt.title(title)
    plt.show()


def birkhoff_decomposition(matrix):
    if not is_balanced(matrix):
        return None, "Graph is not balanced Birkhoff algorithm cannot proceed."

    n = matrix.shape[0]
    working_matrix = matrix.copy()
    steps = []


    j = 1  # Initiali: j = 1

    while True:
        # Step 2:find perfect matching X_j in G
        G = nx.Graph()
        for i in range(n):
            for j_col in range(n):
                if working_matrix.iloc[i, j_col] > 1e-8:
                    G.add_edge(f"P{i}", f"I{j_col}")

        matching_raw = maximum_matching(G, top_nodes={f"P{i}" for i in range(n)})
        X_j = [(int(p[1:]), int(i[1:])) for p, i in matching_raw.items() if p.startswith("P")]

        # Only keep edges that exist in current matrix (non-zero weight)
        X_j = [(i, j_col) for (i, j_col) in X_j if working_matrix.iloc[i, j_col] > 1e-8]

        if len(X_j) < n:
            return steps, "No perfect matching — algorithm stopped."

        # Step 3:find p_j = min weight in X_j
        p_j = min(working_matrix.iloc[i, j_col] for i, j_col in X_j)

        # Step 4:subtract p_j from each edge in X_j, remove edge if weight becomes 0
        for (i, j_col) in X_j:
            working_matrix.iloc[i, j_col] -= p_j
            if working_matrix.iloc[i, j_col] < 1e-8:
                working_matrix.iloc[i, j_col] = 0.0

        step = {
            "Step": j,
            "Matching": [(working_matrix.index[i], working_matrix.columns[j_col]) for i, j_col in X_j],
            "Weight": p_j
        }
        steps.append(step)

        draw_graph(working_matrix, f"After Step j = {j} (Removed p_j = {p_j})")

        # Step 5:if graph is empty, stop; otherwise increase j and continue
        if (working_matrix > 1e-8).sum().sum() == 0:
            break

        j += 1

    draw_graph(working_matrix, "Final Graph (Empty)")
    return steps, "Birkhoff decomposition successful."


if __name__ == "__main__":
    n = 4
    matrix = generate_balanced_matrix(n)
    print("Initial matrix:")
    print(matrix.round(2))
    draw_graph(matrix, "Initial Matrix")
    steps, message = birkhoff_decomposition(matrix)
    print("\nResult:")
    if steps is not None:
        for step in steps:
            print(f"Step {step['Step']}: {step['Matching']} (Weight {step['Weight']})")
    print("\n" + message)
