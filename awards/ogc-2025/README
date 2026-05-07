---

## Overview

This page organizes the code and files for **OGC 2025 / OptiGrandChallenge**.

The uploaded files describe a graph-based routing and allocation problem where each solution is represented as port-indexed routes for loading, unloading, and rehandling operations. The repository structure below is designed so that the award entry in the main `README.md` can be clicked and reviewed directly on GitHub.

---

## Repository Map

| Category | Files | Description |
|---|---|---|
| Main algorithm | [`myalgorithm_final.py`](src/myalgorithm_final.py) | Primary submitted/working algorithm file organized for GitHub. |
| Algorithm variant | [`myalgorithm_variant.py`](src/myalgorithm_variant.py) | Additional algorithm version retained for comparison. |
| Utility module | [`util.py`](src/util.py) | Feasibility checking, BFS, Dijkstra, and path-backtracking utilities. |
| Visualization scripts | [`visual_grid.py`](src/visual_grid.py), [`visual_results.py`](src/visual_results.py) | Plotly-based 3D grid and result-state visualization scripts. |
| Test instances | [`test_prob1.json`](data/test_prob1.json), [`test_prob2.json`](data/test_prob2.json), [`test_prob3.json`](data/test_prob3.json), [`test_prob4.json`](data/test_prob4.json), [`test_prob5.json`](data/test_prob5.json) | Problem input instances. |
| Result file | [`results_prob5.json`](results/results_prob5.json) | Uploaded result JSON. |
| HTML visualizations | [`prob1.html`](visualizations/prob1.html), [`prob2.html`](visualizations/prob2.html), [`prob3.html`](visualizations/prob3.html), [`prob4.html`](visualizations/prob4.html), [`prob5.html`](visualizations/prob5.html) | Pre-generated Plotly HTML files. |
| File manifest | [`MANIFEST.md`](MANIFEST.md) | Original upload names and renamed GitHub paths. |

---

## Problem Instance Summary

| Instance | Nodes `N` | Edges `E` | Ports `P` | Demands `K` | Fixed Cost `F` | Lower Bound `LB` |
|---|---:|---:|---:|---:|---:|---:|
| prob1 | 77 | 106 | 7 | 13 | 100 | 28117 |
| prob2 | 89 | 122 | 10 | 23 | 100 | 32138 |
| prob3 | 85 | 113 | 10 | 13 | 100 | 37882 |
| prob4 | 162 | 225 | 12 | 34 | 100 | 77361 |
| prob5 | 279 | 399 | 10 | 25 | 100 | 107637 |

---

## Uploaded Result Summary

| Field | Value |
|---|---:|
| Problem | `prob5` |
| Feasible | `True` |
| Objective | `5843.0` |
| Runtime | `57.62s` |
| Timelimit exception | `False` |
| Total routes in uploaded solution | `1015` |

The objective and feasibility values are taken from the uploaded `results_prob5.json` file. They are not recomputed in this README.

---

## Algorithm Design

<table>
  <tr>
    <td><b>Path Search</b></td>
    <td>Occupancy-aware shortest path search with NetworkX and cache-based path lookup.</td>
  </tr>
  <tr>
    <td><b>Loading</b></td>
    <td>Heuristic placement using distance, blocking, passageway, and destination-related scoring.</td>
  </tr>
  <tr>
    <td><b>Unloading</b></td>
    <td>Cluster/frontier-style unloading with blocker handling and rehandling decisions.</td>
  </tr>
  <tr>
    <td><b>Rehandling</b></td>
    <td>Compares relocation and temporary unloading/reloading strategies under route-cost logic.</td>
  </tr>
  <tr>
    <td><b>Validation</b></td>
    <td>Utility checker validates route structure, blocked paths, valid edges, demand status, and objective calculation.</td>
  </tr>
</table>

---

## How to Run Locally

Install dependencies:

```bash
pip install -r awards/ogc-2025/requirements.txt
```

Generate a grid visualization from a problem instance:

```bash
python awards/ogc-2025/src/visual_grid.py awards/ogc-2025/data/test_prob1.json
```

Generate a result-state visualization using the uploaded result file:

```bash
python awards/ogc-2025/src/visual_results.py awards/ogc-2025/data/test_prob5.json awards/ogc-2025/results/results_prob5.json
```

For the algorithm itself, the official evaluator should call:

```python
algorithm(prob_info, timelimit=60)
```

The HTML files under [`visualizations/`](visualizations/) are static Plotly exports. On GitHub, they may open as source code rather than as interactive pages. For interactive viewing, open them locally in a browser or serve them through GitHub Pages.

---

<div align="center">

[Back to Profile](../../README.md)

</div>
