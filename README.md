# Birkhoff Decomposition Algorithm

This Python script implements the **Birkhoff–von Neumann decomposition algorithm** using `networkx` and `matplotlib` for visualization.

The algorithm decomposes a **doubly stochastic matrix** (a balanced square matrix where each row and column sums to 1) into a convex combination of **permutation matrices**, using iterative perfect matchings.

---

## 🔧 Features

- Generate random balanced matrices of size `n × n`
- Visualize the matrix as a **bipartite graph**
- Perform Birkhoff decomposition step-by-step:
  - Find a perfect matching \( X_j \)
  - Find minimum edge weight \( p_j \)
  - Subtract \( p_j \) from all edges in the matching
  - Remove zero-weight edges
- Visual feedback via updated graphs per step
- Full logging of matchings and weights


