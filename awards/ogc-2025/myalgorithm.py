

import json
import time
import util
import networkx as nx
from itertools import islice
import numpy as np


def algorithm(prob_info, timelimit=60):
    """
    This is a template for the custom algorithm.
    The algorithm should return a solution that is a list of (route, k) pairs.
    Each route is a list of node indices, and k is the index of the demand that is moved by the route.
    You CANNOT change or remove this function signature.
    But it is fine to define extra functions or mudules that are used in this function.
    """




    #------------- begin of custom algorithm code --------------#
time_limit_buffer = 5
USE_BFS_FRONTIER_FOR_UNLOAD = True

import time
from functools import lru_cache
from dataclasses import dataclass

# =========================
# 0) Scoring config & utils
# =========================
@dataclass
class ScoringConfig:
    conflict_N_threshold: int = 300
    dist2_weight: float = 0.5
    use_bfs_frontier_for_unload: bool = True

    same_dest_reward: float = 44.31171074572622
    conflict_penalty: float = 100.0
    passageway_penalty: float = 50.81053317715756
    blocked_penalty_factor: float = 148.19469253211534
    distance_weight: float = 0.9171485191417306
    explore_noise_range_far: float = 25.0
    explore_noise_range_flex: float = 10.773510669976911

    rehandle_same_dest_reward: float = 50.0
    rehandle_passageway_penalty: float = 50.0
    rehandle_blocked_penalty_factor: float = 150.0
    rehandle_distance_weight: float = 1.0

GLOBAL_CFG = globals().get("GLOBAL_CFG") or ScoringConfig()

GLOBAL_PHASE1_HEUR = None  # will be bound inside algorithm() for initial-seed delegation

def get_global_params():
    return {
        "conflict_N_threshold": GLOBAL_CFG.conflict_N_threshold,
        "dist2_weight": GLOBAL_CFG.dist2_weight,
        "use_bfs_frontier_for_unload": GLOBAL_CFG.use_bfs_frontier_for_unload,
        "same_dest_reward": GLOBAL_CFG.same_dest_reward,
        "conflict_penalty": GLOBAL_CFG.conflict_penalty,
        "passageway_penalty": GLOBAL_CFG.passageway_penalty,
        "blocked_penalty_factor": GLOBAL_CFG.blocked_penalty_factor,
        "distance_weight": GLOBAL_CFG.distance_weight,
        "explore_noise_range_far": GLOBAL_CFG.explore_noise_range_far,
        "explore_noise_range_flex": GLOBAL_CFG.explore_noise_range_flex,
        "rehandle_same_dest_reward": GLOBAL_CFG.rehandle_same_dest_reward,
        "rehandle_passageway_penalty": GLOBAL_CFG.rehandle_passageway_penalty,
        "rehandle_blocked_penalty_factor": GLOBAL_CFG.rehandle_blocked_penalty_factor,
        "rehandle_distance_weight": GLOBAL_CFG.rehandle_distance_weight,
    }

def set_global_params(overrides: dict):
    for k, v in (overrides or {}).items():
        if hasattr(GLOBAL_CFG, k):
            setattr(GLOBAL_CFG, k, v)
    return get_global_params()
# ===========================
# Scoring schedule & logging
# ===========================
GLOBAL_SCORING_LOG = []           # list of (iteration, params dict)
BEST_SCORING_AT_BEST = None       # params at time best_solution updated
BEST_SCORING_ITER = None          # iteration when best accepted
GLOBAL_SCORING_BASE = None         # baseline params at init

def _record_scoring(iteration_label):
    try:
        GLOBAL_SCORING_LOG.append((iteration_label, dict(get_global_params())))
    except Exception:
        pass

def _apply_scoring_schedule(iteration_count:int, total_allow:int=GLOBAL_CFG.conflict_penalty):
    """Anchor to baseline so changes are bounded (no compounding)."""
    base = globals().get('GLOBAL_SCORING_BASE') or get_global_params()
    prog = min(1.0, max(0.0, iteration_count / float(max(1,total_allow))))
    amp = 0.3 * prog  # up to +30%

    new_params = {
        "same_dest_reward": base["same_dest_reward"] * (1.0 + amp),
        "passageway_penalty": base["passageway_penalty"] * (1.0 + amp),
        "blocked_penalty_factor": base["blocked_penalty_factor"] * (1.0 + amp),
        "distance_weight": base["distance_weight"] * (1.0 - 0.2 * prog),
    }
    set_global_params(new_params)


# --- Module-level time guards (external only; 비용 함수 내부엔 미사용) ---
GLOBAL_START = None
GLOBAL_TIMELIMIT = None
SAFETY_BUFFER_SEC = 5  # seconds

def _time_left():
    if GLOBAL_TIMELIMIT is None:
        return float('inf')
    return (GLOBAL_START + GLOBAL_TIMELIMIT) - time.time()

# =============================
# 1) Graph helpers & path cache
# =============================
import itertools
import util
import networkx as nx
from itertools import islice
import numpy as np
from types import SimpleNamespace

# Graph registry for cached funcs
_PATH_GRAPH_REGISTRY = {}

class LazyShortestPaths:
    def __init__(self, G, source, max_k):
        self.G = G
        self.source = source
        self.max_k = max_k
        self._cache = {}
    def __getitem__(self, target):
        if target in self._cache:
            return self._cache[target]
        try:
            paths = list(itertools.islice(nx.shortest_simple_paths(self.G, self.source, target), self.max_k))
        except nx.NetworkXNoPath:
            paths = []
        self._cache[target] = paths
        return paths

class PathFinder:
    """Occupancy-aware shortest path with LRU cache."""
    def __init__(self, G):
        self.G = G
        self.graph_id = id(G)
        _PATH_GRAPH_REGISTRY[self.graph_id] = G

    @staticmethod
    @lru_cache(maxsize=4096)
    def _cached(graph_id, start_node, end_node, occ_tuple):
        G = _PATH_GRAPH_REGISTRY.get(graph_id)
        if G is None:
            return None
        occupied = set(occ_tuple)
        allowed = lambda n: (n not in occupied) or (n == start_node) or (n == end_node)
        H = nx.subgraph_view(G, filter_node=allowed)
        try:
            return nx.shortest_path(H, source=start_node, target=end_node)
        except nx.NetworkXNoPath:
            return None

    def find(self, start_node, end_node, node_allocations):
        try:
            import numpy as _np
            occ_tuple = tuple(map(int, _np.flatnonzero(_np.asarray(node_allocations) != -1)))
        except Exception:
            occ_tuple = tuple(i for i,k in enumerate(node_allocations) if k != -1)
        return self._cached(self.graph_id, start_node, end_node, occ_tuple)

# Pairwise shortest path length cache
@lru_cache(maxsize=100000)
def _sp_len_cached(graph_id, u, v, weighted_flag):
    G = _PATH_GRAPH_REGISTRY.get(graph_id)
    if G is None:
        return None
    try:
        if weighted_flag:
            return nx.shortest_path_length(G, u, v, weight='weight')
        else:
            return nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        return None

def shortest_path_length_cached(G, u, v, weight='weight'):
    weighted_flag = (weight is not None and weight is not False)
    gid = id(G)
    _PATH_GRAPH_REGISTRY[gid] = G
    return _sp_len_cached(gid, u, v, weighted_flag)

# Pairwise k-shortest paths cache
@lru_cache(maxsize=50000)
def _ksp_cached(graph_id, src, dst, k):
    G = _PATH_GRAPH_REGISTRY.get(graph_id)
    if G is None:
        return []
    try:
        return list(itertools.islice(nx.shortest_simple_paths(G, src, dst), k))
    except nx.NetworkXNoPath:
        return []

def k_shortest_paths_cached(G, src, dst, k):
    gid = id(G)
    _PATH_GRAPH_REGISTRY[gid] = G
    return _ksp_cached(gid, src, dst, int(k))

# Pretty JSON (optional)
try:
    import jsbeautifier as _jsb
    def _beautify_json(s):
        opts = _jsb.default_options(); opts.indent_size = 2
        return _jsb.beautify(s, opts)
except Exception:
    def _beautify_json(s):
        return s

# ================
# 2) Global caches
# ================
NEI1 = None
NEI2 = None
IS_PASS = None
GATE_PARENT = None
MAX_CAND = 256  # soft cap for candidate sets

# Worker/global placeholders
_WORKER_CTX = None
_GLOBAL_G = None
_CURRENT_PATHFINDER = None

# ==================
# 3) Safe backtrack
# ==================
def _backtrack_safe(previous_nodes, start, target):
    """Robust backtracking even if 'start' not explicitly present."""
    if previous_nodes is None:
        return None
    prev = dict(previous_nodes)
    if start not in prev:
        prev[start] = None
    curr = target
    seen = set()
    while curr is not None and curr != start:
        if curr in seen:
            return None
        seen.add(curr)
        if curr not in prev:
            return None
        curr = prev[curr]
    if curr != start:
        return None
    try:
        _p = util.path_backtracking(prev, start, target)
        return list(_p) if _p is not None else None
    except KeyError:
        return None

# ============================
# 4) Cost & scoring primitives
# ============================
@lru_cache(maxsize=100000)
def _cost_port_cached(signature, F_int):
    total = 0
    for route, k in signature:
        if len(route) < 2:
            return float('inf')
        total += F_int + (len(route) - 1)
    return total

def calculate_total_cost(solution, P, F):
    """전체 비용(포트별 고정비+이동거리). 정확성을 위해 타임가드 없음."""
    if solution is None:
        return float('inf')
    total_cost = 0
    F_int = int(F)
    for p in range(P):
        routes = solution.get(p, [])
        try:
            signature = tuple((tuple(route), int(k)) for route, k in routes)
        except Exception:
            signature = tuple((tuple(route), k) for route, k in routes)
        port_cost = _cost_port_cached(signature, F_int)
        if port_cost == float('inf'):
            return float('inf')
        total_cost += port_cost
    return total_cost

# ==================================
# 5) Path helpers (occupancy-aware)
# ==================================
# find_path, rehandling, unloading, loading heuristics from 쌀먹.py will be inserted here.

# --- (신규) 쌀먹.py에서 가져온 함수 시작 ---

def find_path(start_node, end_node, node_allocations, G):
    # Use cached PathFinder if bound to this graph
    if '_CURRENT_PATHFINDER' in globals():
        pf = globals().get('_CURRENT_PATHFINDER', None)
        if pf is not None and getattr(pf, 'G', None) is G:
            out = pf.find(start_node, end_node, node_allocations)
            if out is not None:
                return out
    """Optimized: build a subgraph view that excludes occupied nodes (except endpoints)."""
    allowed = set(
        i for i, k_idx in enumerate(node_allocations)
        if (k_idx == -1) or (i == start_node) or (i == end_node)
    )
    H = nx.subgraph_view(G, filter_node=lambda n: n in allowed)
    try:
        return nx.shortest_path(H, source=start_node, target=end_node)
    except nx.NetworkXNoPath:
        return None

def find_best_rehandling_location(node_allocations, G, nodes_to_avoid=None):
    reachable_nodes, _ = util.bfs(G, node_allocations)
    empty_nodes = [n for n in reachable_nodes if node_allocations[n] == -1 and n != 0]
    if nodes_to_avoid:
        avoid_set = set(nodes_to_avoid)
        empty_nodes = [n for n in empty_nodes if n not in avoid_set]
    if not empty_nodes: return None
    best_node, max_distance = -1, -1
    distances, _ = util.dijkstra(G, node_allocations)
    for node in empty_nodes:
        distance = distances.get(node, -1)
        if distance > max_distance:
            max_distance, best_node = distance, node
    return best_node

def find_best_location_by_score_for_rehandling(k_to_place, node_allocations, K, G, passageway_nodes, nodes_to_avoid=None):
    reachable_nodes, reachable_node_distances_list = util.bfs(G, node_allocations)
    empty_nodes = [n for n in reachable_nodes if node_allocations[n] == -1 and n != 0]
    if nodes_to_avoid:
        avoid_set = set(nodes_to_avoid)
        empty_nodes = [n for n in empty_nodes if n not in avoid_set]
    if not empty_nodes: return None
    reachable_node_distances = {node: dist for node, dist in zip(reachable_nodes, reachable_node_distances_list)}
    best_node, best_score = -1, -float('inf')
    dest_to_place = K[k_to_place][0][1]
    for n_candidate in empty_nodes:
        current_score = 0
        neighbors = set(G.neighbors(n_candidate))
        for neighbor in neighbors:
            if node_allocations[neighbor] != -1:
                neighbor_k = int(node_allocations[neighbor])
                neighbor_dest = K[neighbor_k][0][1]
                current_score += (neighbor_dest - dest_to_place)
                if dest_to_place == neighbor_dest: current_score += 50
        if n_candidate in passageway_nodes: current_score -= 50
        hypothetical_allocations = node_allocations.copy()
        hypothetical_allocations[n_candidate] = k_to_place
        reachable_after, _ = util.bfs(G, hypothetical_allocations)
        num_blocked = len(set(empty_nodes) - {n_candidate} - set(reachable_after))
        current_score -= num_blocked * 150
        current_score += reachable_node_distances.get(n_candidate, 0)
        if current_score > best_score:
            best_score, best_node = current_score, n_candidate
    return best_node

def intelligent_rehandling_decider(p, blocking_node, node_allocations, K, P, F, G, passageway_nodes, shortest_paths, path_to_clear=None):
    k_block = int(node_allocations[blocking_node])
    relocation_cost = float('inf')
    best_relocation_path = None
    if p < P - 1:
        relocation_target = find_best_location_by_score_for_rehandling(
            k_block, node_allocations, K, G, passageway_nodes, nodes_to_avoid=path_to_clear
        )
        if relocation_target:
            relocation_path = find_path(blocking_node, relocation_target, node_allocations, G)
            if relocation_path:
                relocation_cost = F + len(relocation_path) - 1
                best_relocation_path = relocation_path
    temp_unload_path = find_path(blocking_node, 0, node_allocations, G)
    total_temp_cost = float('inf')
    if temp_unload_path:
        temp_unload_cost = F + len(temp_unload_path) - 1
        if p == P - 1:
            total_temp_cost = temp_unload_cost
        else:
            temp_allocs = node_allocations.copy()
            temp_allocs[blocking_node] = -1
            reload_target = find_best_rehandling_location(temp_allocs, G, nodes_to_avoid=path_to_clear)
            if reload_target:
                reload_path = find_path(0, reload_target, temp_allocs, G)
                if reload_path:
                    total_temp_cost = temp_unload_cost + F + len(reload_path) - 1
    rehandling_routes, rehandling_demands, updated_allocations = [], [], node_allocations.copy()
    if relocation_cost < total_temp_cost:
        rehandling_routes.append((best_relocation_path, k_block))
        updated_allocations[best_relocation_path[-1]] = k_block
        updated_allocations[blocking_node] = -1
    else:
        if temp_unload_path:
            rehandling_routes.append((temp_unload_path, k_block))
            if p < P - 1:
                rehandling_demands.append(k_block)
            updated_allocations[blocking_node] = -1
        else: return None, None, None
    return rehandling_routes, rehandling_demands, updated_allocations

def unloading_heuristic(p, node_allocations, K, P, F, shortest_paths, G, passageway_nodes):
    route_list, rehandling_demands = [], []
    alloc = node_allocations.copy()
    K_unload_set = {idx for idx, ((o, d), r) in enumerate(K) if d == p}

    def occ_nodes_for_p():
        return {int(n) for n, k in enumerate(alloc) if k != -1 and int(k) in K_unload_set}

    def find_seed_by_bfs():
        dist, prev = util.dijkstra(G, alloc, start=0)
        candidates = sorted([n for n, d in dist.items() if n != 0 and d < float('inf')], key=lambda x: dist[x])
        for r in candidates:
            for v in G[r]:
                if v in occ_nodes_for_p():
                    path0_to_r = _backtrack_safe(prev, 0, r)
                    if path0_to_r: return v, path0_to_r + [v]
        return None, None
    def find_seed_by_min_resistance():
        import networkx as _nx
        if not USE_BFS_FRONTIER_FOR_UNLOAD:
            # Baseline: single shortest path 0->seed with minimum blockers
            best = (float('inf'), float('inf'), None, None, None)
            for n in occ_nodes_for_p():
                try:
                    sp = _nx.shortest_path(G, source=0, target=n)
                except _nx.NetworkXNoPath:
                    continue
                interior = sp[1:-1]
                blockers_list = [x for x in interior if alloc[x] != -1]
                cand = (len(blockers_list), len(sp), n, sp, blockers_list)
                if cand < best:
                    best = cand
            if best[2] is None:
                return None, None, None
            return best[2], best[3], best[4]

        # BFS-frontier + min-blockers connect (feature flag)
        from collections import deque
        import heapq

        # BFS over free nodes from gate
        R = set([0])
        parent_gate = {0: None}
        dq = deque([0])
        while dq:
            v = dq.popleft()
            for w in G.neighbors(v):
                if w in R: 
                    continue
                if alloc[w] == -1:
                    R.add(w); parent_gate[w] = v; dq.append(w)
        if not R:
            return None, None, None

        def reconstruct_gate_path(x):
            path = []
            cur = x
            while cur is not None:
                path.append(cur)
                cur = parent_gate.get(cur)
            path.reverse()
            return path

        best = (float('inf'), float('inf'), None, None, None)
        for seed in list(occ_nodes_for_p()):
            pq = [(0, 0, seed)]
            seen = {}
            parent = {seed: None}
            meet_node = None

            while pq:
                blk, dist, v = heapq.heappop(pq)
                prev = seen.get(v)
                if prev is not None and (blk, dist) > prev:
                    continue
                seen[v] = (blk, dist)
                if v in R:
                    meet_node = v
                    break
                for w in G.neighbors(v):
                    nb_blk = blk + (1 if (alloc[w] != -1) else 0)
                    nb_dist = dist + 1
                    prev2 = seen.get(w)
                    if prev2 is None or (nb_blk, nb_dist) < prev2:
                        seen[w] = (nb_blk, nb_dist)
                        parent[w] = v
                        heapq.heappush(pq, (nb_blk, nb_dist, w))

            if meet_node is None:
                continue

            # Reconstruct seed->meet, then full path 0..meet..seed
            path_seed_to_meet = []
            cur = meet_node
            while cur is not None:
                path_seed_to_meet.append(cur)
                cur = parent.get(cur)
            path_seed_to_meet.reverse()
            gate_part = reconstruct_gate_path(meet_node)
            meet_to_seed = list(reversed(path_seed_to_meet))  # meet..seed
            full_path = gate_part + meet_to_seed[1:]

            interior = full_path[1:-1]
            blockers_list = [x for x in interior if alloc[x] != -1]
            cand = (len(blockers_list), len(full_path), seed, full_path, blockers_list)
            if cand < best:
                best = cand

        if best[2] is None:
            return None, None, None
        return best[2], best[3], best[4]

    def handle_blockers_intelligently(corridor, blockers):
        nonlocal alloc, route_list, rehandling_demands
        idx = {node: i for i, node in enumerate(corridor)}
        blockers_sorted = sorted(blockers, key=lambda b: idx.get(b, float('inf')))
        for bn in blockers_sorted:
            if alloc[bn] == -1: continue
            
            rehandled_routes, demands, updated_allocs = intelligent_rehandling_decider(
                p, bn, alloc, K, P, F, G, passageway_nodes, shortest_paths, path_to_clear=corridor
            )
            if rehandled_routes is None: return False
            
            route_list.extend(rehandled_routes)
            rehandling_demands.extend(demands)
            alloc = updated_allocs
        return True

    def process_cluster(seed_node, corridor):
        nonlocal alloc, route_list
        k_seed = int(alloc[seed_node])
        route_list.append((corridor[::-1], k_seed)); alloc[seed_node] = -1
        
        # BFS/DFS-like traversal from the seed to find connected unloadable items
        from collections import deque
        q = deque([seed_node])
        processed_in_cluster = {seed_node}
        
        while q:
            u = q.popleft()
            for v in G[u]:
                if v in occ_nodes_for_p() and v not in processed_in_cluster:
                    processed_in_cluster.add(v)
                    k_v = int(alloc[v])
                    path_to_gate = find_path(v, 0, alloc, G)
                    if path_to_gate:
                        route_list.append((path_to_gate, k_v))
                        alloc[v] = -1
                        q.append(v)

    while True:
        if not occ_nodes_for_p(): break
        seed, corridor, blockers = None, None, []
        seed, corridor = find_seed_by_bfs()
        if seed is None:
            seed, corridor, blockers = find_seed_by_min_resistance()
            if seed is None: break
        
        if blockers:
            ok = handle_blockers_intelligently(corridor, blockers)
            if not ok: break
        
        process_cluster(seed, corridor)
        
    return route_list, rehandling_demands, alloc

def loading_heuristic_farthest_first(p, node_allocations, demands_to_load, G, passageway_nodes, K, is_exploring=True):
    import random
    import util
    """가장 먼 노드부터 채우는 상하차 휴리스틱입니다. O-D 거리가 1이 아닌 수요를 먼저 처리하고, O-D 거리가 1인 수요를 게이트 가까운 순서로 처리합니다. N >= 300일 때 충돌 패널티를 적용합니다."""
    route_list = []
    current_allocations = node_allocations.copy()

    total_loading_demands_count = sum(demands_to_load.values())
    if total_loading_demands_count == 0:
        return [], current_allocations

    N = len(node_allocations)

    def _check_conflict(K_info, demand_k1, demand_k2):
        o1, d1 = K_info[demand_k1][0]
        o2, d2 = K_info[demand_k2][0]
        return (o1 < o2 and d1 > o2 and d1 < d2) or \
               (o2 < o1 and d2 > o1 and d2 < d1)

    demands_not_dist_one = {}
    demands_dist_one = {}

    # During initial solution (is_exploring == False), disable short-term (|d-o|==1) special handling
    ignore_short_term = (is_exploring is False)
    for k_idx, quantity in demands_to_load.items():
        o_port, d_port = K[k_idx][0]
        if ignore_short_term:
            demands_not_dist_one[k_idx] = quantity
        else:
            if abs(d_port - o_port) == 1:
                demands_dist_one[k_idx] = quantity
            else:
                demands_not_dist_one[k_idx] = quantity
    
    reachable_nodes, reachable_node_distances_list = util.bfs(G, current_allocations)
    if len(reachable_nodes) < total_loading_demands_count:
        return None, None

    reachable_node_distances = {node: dist for node, dist in zip(reachable_nodes, reachable_node_distances_list)}
    
    loading_nodes_pool = [n for n in reachable_nodes if n != 0]

    if len(loading_nodes_pool) < total_loading_demands_count:
        return None, None

    sorted_K_load_for_current_phase = sorted(demands_not_dist_one.items(), key=lambda x: K[x[0]][0][1], reverse=True)
    flattened_K_load_for_current_phase = [k for k, r in sorted_K_load_for_current_phase for _ in range(r)]

    non_passageway_reachable_current = sorted([n for n in loading_nodes_pool if n not in passageway_nodes], key=lambda n: reachable_node_distances.get(n, 0), reverse=True)
    passageway_reachable_current = sorted([n for n in loading_nodes_pool if n in passageway_nodes], key=lambda n: reachable_node_distances.get(n, 0), reverse=True)
    candidate_nodes_current_phase = non_passageway_reachable_current + passageway_reachable_current

    for k_to_place in flattened_K_load_for_current_phase:
        dest_to_place = K[k_to_place][0][1]

        if not candidate_nodes_current_phase:
            return None, None

        scored = []
        for n_candidate in candidate_nodes_current_phase:
            if current_allocations[n_candidate] != -1:
                continue
            current_score = 0
            neighbors_dist1 = set(G.neighbors(n_candidate))
            for neighbor in neighbors_dist1:
                if current_allocations[neighbor] != -1:
                    neighbor_k = current_allocations[neighbor]
                    neighbor_dest = K[neighbor_k][0][1]
                    current_score += (neighbor_dest - dest_to_place)
                    if dest_to_place == neighbor_dest:
                        current_score += GLOBAL_CFG.same_dest_reward
                    if N >= GLOBAL_CFG.conflict_N_threshold and _check_conflict(K, k_to_place, neighbor_k):
                        current_score -= GLOBAL_CFG.conflict_penalty
            neighbors_dist2 = set()
            for n1 in neighbors_dist1:
                for n2 in G.neighbors(n1):
                    if n2 != n_candidate and n2 not in neighbors_dist1:
                        neighbors_dist2.add(n2)
            dist2_weight = GLOBAL_CFG.dist2_weight
            for neighbor in neighbors_dist2:
                if current_allocations[neighbor] != -1:
                    neighbor_k = current_allocations[neighbor]
                    neighbor_dest = K[neighbor_k][0][1]
                    current_score += (neighbor_dest - dest_to_place) * dist2_weight
                    if dest_to_place == neighbor_dest:
                        current_score += GLOBAL_CFG.same_dest_reward * dist2_weight
                    if N >= GLOBAL_CFG.conflict_N_threshold and _check_conflict(K, k_to_place, neighbor_k):
                        current_score -= GLOBAL_CFG.conflict_penalty * dist2_weight
            if n_candidate in passageway_nodes:
                current_score -= GLOBAL_CFG.passageway_penalty
            hypothetical_allocations = current_allocations.copy()
            hypothetical_allocations[n_candidate] = k_to_place
            reachable_after, _ = util.bfs(G, hypothetical_allocations)
            num_blocked = len(set(candidate_nodes_current_phase) - {n_candidate} - set(reachable_after))
            current_score -= num_blocked * GLOBAL_CFG.blocked_penalty_factor
            current_score += reachable_node_distances.get(n_candidate, 0) * GLOBAL_CFG.distance_weight
            if is_exploring: current_score += random.uniform(-GLOBAL_CFG.explore_noise_range_flex, GLOBAL_CFG.explore_noise_range_flex)
            scored.append((current_score, n_candidate))

        scored.sort(reverse=True)
        placed = False
        distances, previous_nodes = util.dijkstra(G, current_allocations)
        for _, n in scored:
            if n not in previous_nodes or distances[n] == float('inf'):
                continue
            path = _backtrack_safe(previous_nodes, 0, n)
            if not path:
                continue
            route_list.append((path, k_to_place))
            current_allocations[n] = k_to_place
            if n in loading_nodes_pool:
                loading_nodes_pool.remove(n)
            if n in candidate_nodes_current_phase:
                candidate_nodes_current_phase.remove(n)
            placed = True
            break
        if not placed:
            print('[DEBUG] all top candidates unreachable for k', k_to_place)
            return None, None

    if demands_dist_one:
        sorted_demands_dist_one = sorted(demands_dist_one.items(), key=lambda x: K[x[0]][0][1], reverse=True)
        flattened_demands_dist_one = [k for k, r in sorted_demands_dist_one for _ in range(r)]
        num_demands_dist_one = len(flattened_demands_dist_one)

        if num_demands_dist_one > 0:
            remaining_reachable_nodes_for_dist_one = [n for n in loading_nodes_pool if current_allocations[n] == -1]
            sorted_nodes_by_distance = sorted(remaining_reachable_nodes_for_dist_one, key=lambda n: reachable_node_distances.get(n, float('inf')))

            if len(sorted_nodes_by_distance) < num_demands_dist_one:
                return None, None
            
            nodes_to_assign_dist_one = sorted_nodes_by_distance[:num_demands_dist_one]

            for i, k_to_place in enumerate(flattened_demands_dist_one):
                if not nodes_to_assign_dist_one:
                    return None, None
                
                target_node = nodes_to_assign_dist_one[num_demands_dist_one - 1 - i]
                
                distances, previous_nodes = util.dijkstra(G, current_allocations)
                if target_node == -1 or target_node not in previous_nodes or distances[target_node] == float('inf'):
                    # Fallback: pick next closest available node with valid path
                    cand_nodes = [n for n in sorted_nodes_by_distance if current_allocations[n] == -1]
                    picked = None
                    for n2 in cand_nodes:
                        d2, prev2 = util.dijkstra(G, current_allocations)
                        p2 = _backtrack_safe(prev2, 0, n2)
                        if p2 is not None and d2.get(n2, float('inf')) < float('inf'):
                            picked = (n2, p2)
                            break
                    if picked is None:
                        return None, None
                    target_node, path = picked
                else:
                    path = _backtrack_safe(previous_nodes, 0, target_node)
                    if path is None:
                        return None, None
                
                path = _backtrack_safe(previous_nodes, 0, target_node)
                route_list.append((path, k_to_place))
                current_allocations[target_node] = k_to_place
                
                if target_node in loading_nodes_pool:
                    loading_nodes_pool.remove(target_node)
                
    return route_list, current_allocations

def loading_heuristic_flexible_score(p, node_allocations, demands_to_load, G, passageway_nodes, K, is_exploring=True):
    import random
    import util
    cfg = GLOBAL_CFG  # use runtime-configurable scoring
    """유연한 점수 계산 방식의 상하차 휴리스틱입니다. O-D 거리가 1이 아닌 수요를 먼저 처리하고, O-D 거리가 1인 수요를 게이트 가까운 순서로 처리합니다. N >= 300일 때 충돌 패널티를 적용합니다."""
    route_list = []
    current_allocations = node_allocations.copy()

    total_loading_demands_count = sum(demands_to_load.values())
    if total_loading_demands_count == 0:
        return [], current_allocations

    N = len(node_allocations)

    def _check_conflict(K_info, demand_k1, demand_k2):
        o1, d1 = K_info[demand_k1][0]
        o2, d2 = K_info[demand_k2][0]
        return (o1 < o2 and d1 > o2 and d1 < d2) or \
               (o2 < o1 and d2 > o1 and d2 < d1)

    demands_not_dist_one = {}
    demands_dist_one = {}

    for k_idx, quantity in demands_to_load.items():
        o_port, d_port = K[k_idx][0]
        if abs(d_port - o_port) == 1:
            demands_dist_one[k_idx] = quantity
        else:
            demands_not_dist_one[k_idx] = quantity
    
    reachable_nodes, reachable_node_distances_list = util.bfs(G, current_allocations)
    if len(reachable_nodes) < total_loading_demands_count:
        return None, None

    reachable_node_distances = {node: dist for node, dist in zip(reachable_nodes, reachable_node_distances_list)}
    
    loading_nodes_pool = [n for n in reachable_nodes if n != 0]

    if len(loading_nodes_pool) < total_loading_demands_count:
        return None, None

    sorted_K_load_for_current_phase = sorted(demands_not_dist_one.items(), key=lambda x: K[x[0]][0][1], reverse=True)
    flattened_K_load_for_current_phase = [k for k, r in sorted_K_load_for_current_phase for _ in range(r)]

    non_passageway_reachable = [n for n in loading_nodes_pool if n not in passageway_nodes]
    passageway_reachable = [n for n in loading_nodes_pool if n in passageway_nodes]
    candidate_nodes_current_phase = non_passageway_reachable + passageway_reachable

    for k_to_place in flattened_K_load_for_current_phase:
        best_node, best_score = -1, -float('inf')
        dest_to_place = K[k_to_place][0][1]

        if not candidate_nodes_current_phase:
            return None, None
        
        for n_candidate in candidate_nodes_current_phase:
            if current_allocations[n_candidate] != -1:
                continue

            current_score = 0
            neighbors_dist1 = set(G.neighbors(n_candidate))

            for neighbor in neighbors_dist1:
                if current_allocations[neighbor] != -1:
                    neighbor_k = current_allocations[neighbor]
                    neighbor_dest = K[neighbor_k][0][1]
                    current_score += (neighbor_dest - dest_to_place)
                    if dest_to_place == neighbor_dest:
                        current_score += GLOBAL_CFG.same_dest_reward
                    if N >= GLOBAL_CFG.conflict_N_threshold and _check_conflict(K, k_to_place, neighbor_k):
                        current_score -= GLOBAL_CFG.conflict_penalty
            
            neighbors_dist2 = set()
            for n1 in neighbors_dist1:
                for n2 in G.neighbors(n1):
                    if n2 != n_candidate and n2 not in neighbors_dist1:
                        neighbors_dist2.add(n2)

            dist2_weight = GLOBAL_CFG.dist2_weight
            for neighbor in neighbors_dist2:
                if current_allocations[neighbor] != -1:
                    neighbor_k = current_allocations[neighbor]
                    neighbor_dest = K[neighbor_k][0][1]
                    current_score += (neighbor_dest - dest_to_place) * dist2_weight
                    if dest_to_place == neighbor_dest:
                        current_score += GLOBAL_CFG.same_dest_reward * dist2_weight
                    if N >= GLOBAL_CFG.conflict_N_threshold and _check_conflict(K, k_to_place, neighbor_k):
                        current_score -= GLOBAL_CFG.conflict_penalty * dist2_weight

            if n_candidate in passageway_nodes: current_score -= GLOBAL_CFG.passageway_penalty
            
            hypothetical_allocations = current_allocations.copy()
            hypothetical_allocations[n_candidate] = k_to_place
            reachable_after, _ = util.bfs(G, hypothetical_allocations)
            
            num_blocked = len(set(candidate_nodes_current_phase) - {n_candidate} - set(reachable_after))
            current_score -= num_blocked * GLOBAL_CFG.blocked_penalty_factor
            
            current_score += reachable_node_distances.get(n_candidate, 0) * GLOBAL_CFG.distance_weight
            if is_exploring: current_score += random.uniform(-GLOBAL_CFG.explore_noise_range_flex, GLOBAL_CFG.explore_noise_range_flex)
            
            if current_score > best_score:
                best_score, best_node = current_score, n_candidate
        
        distances, previous_nodes = util.dijkstra(G, current_allocations)
        if best_node == -1 or best_node not in previous_nodes or distances[best_node] == float('inf'):
            valid_node = None
            valid_path = None
            for n2 in candidate_nodes_current_phase:
                if current_allocations[n2] != -1:
                    continue
                d2, prev2 = util.dijkstra(G, current_allocations)
                path2 = _backtrack_safe(prev2, 0, n2)
                if path2 is not None and d2.get(n2, float('inf')) < float('inf'):
                    valid_node = n2
                    valid_path = path2
                    break
            if valid_node is None:
                return None, None
            best_node = valid_node
            path = valid_path
        else:
            path = _backtrack_safe(previous_nodes, 0, best_node)
            if path is None:
                return None, None
        
        path = _backtrack_safe(previous_nodes, 0, best_node)
        route_list.append((path, k_to_place))
        current_allocations[best_node] = k_to_place
        
        if best_node in loading_nodes_pool:
            loading_nodes_pool.remove(best_node)
        if best_node in candidate_nodes_current_phase:
            candidate_nodes_current_phase.remove(best_node)

    if demands_dist_one:
        sorted_demands_dist_one = sorted(demands_dist_one.items(), key=lambda x: K[x[0]][0][1], reverse=True)
        flattened_demands_dist_one = [k for k, r in sorted_demands_dist_one for _ in range(r)]
        num_demands_dist_one = len(flattened_demands_dist_one)

        if num_demands_dist_one > 0:
            remaining_reachable_nodes_for_dist_one = [n for n in loading_nodes_pool if current_allocations[n] == -1]
            sorted_nodes_by_distance = sorted(remaining_reachable_nodes_for_dist_one, key=lambda n: reachable_node_distances.get(n, float('inf')))

            if len(sorted_nodes_by_distance) < num_demands_dist_one:
                return None, None
            
            nodes_to_assign_dist_one = sorted_nodes_by_distance[:num_demands_dist_one]

            for i, k_to_place in enumerate(flattened_demands_dist_one):
                if not nodes_to_assign_dist_one:
                    return None, None
                
                target_node = nodes_to_assign_dist_one[num_demands_dist_one - 1 - i]
                
                distances, previous_nodes = util.dijkstra(G, current_allocations)
                if target_node == -1 or target_node not in previous_nodes or distances[target_node] == float('inf'):
                    cand_nodes = [n for n in sorted_nodes_by_distance if current_allocations[n] == -1]
                    picked = None
                    for n2 in cand_nodes:
                        d2, prev2 = util.dijkstra(G, current_allocations)
                        p2 = _backtrack_safe(prev2, 0, n2)
                        if p2 is not None and d2.get(n2, float('inf')) < float('inf'):
                            picked = (n2, p2)
                            break
                    if picked is None:
                        return None, None
                    target_node, path = picked
                else:
                    path = _backtrack_safe(previous_nodes, 0, target_node)
                    if path is None:
                        return None, None
                
                path = _backtrack_safe(previous_nodes, 0, target_node)
                route_list.append((path, k_to_place))
                current_allocations[target_node] = k_to_place
                
                if target_node in loading_nodes_pool:
                    loading_nodes_pool.remove(target_node)
                
    return route_list, current_allocations

def solve_from(start_port, initial_allocations, heuristic_func, P, F, K, shortest_paths, G, passageway_nodes, is_exploring):
    """특정 포트부터 시작하여 솔루션의 일부를 생성합니다."""
    solution_part = {}
    node_allocations = initial_allocations.copy()

    for p in range(start_port, P):
        if _time_left() <= SAFETY_BUFFER_SEC:
            break
            
        solution_part[p] = []
        unload_routes, rehandling_demands, node_allocations = unloading_heuristic(p, node_allocations, K, P, F, shortest_paths, G, passageway_nodes)
        solution_part[p].extend(unload_routes)

        if p < P - 1:
            demands_to_load = {idx: r for idx, ((o, d), r) in enumerate(K) if o == p}
            for k_rehandle in rehandling_demands:
                demands_to_load[k_rehandle] = demands_to_load.get(k_rehandle, 0) + 1
            
            # Note: 쌀먹.py version of heuristic_func does not take 'cfg'
            load_routes, new_allocations = heuristic_func(p, node_allocations, demands_to_load, G, passageway_nodes, K, is_exploring)
            
            if load_routes is None: return None
            
            solution_part[p].extend(load_routes)
            node_allocations = new_allocations
    
    return solution_part

# --- (신규) 쌀먹.py에서 가져온 함수 끝 ---


# =========================================
# 10) Worker pool initializer & repair worker
# =========================================
# _WORKER_CTX: (G, K, P, F, shortest_paths, passageway_nodes, cfg, NEI1, NEI2, IS_PASS, GATE_PARENT)

def _init_pool_worker(ctx_payload):
    global _WORKER_CTX, _CURRENT_PATHFINDER, _GLOBAL_G, NEI1, NEI2, IS_PASS, GATE_PARENT
    _WORKER_CTX = ctx_payload
    G, K, P, F, shortest_paths, passageway_nodes, cfg, nei1, nei2, is_pass, gate_parent = ctx_payload
    _GLOBAL_G = G
    _CURRENT_PATHFINDER = PathFinder(G)
    NEI1 = nei1
    NEI2 = nei2
    IS_PASS = is_pass
    GATE_PARENT = gate_parent

def repair_worker_safe(args):
    try:
        return repair_worker(args)
    except Exception as e:
        try:
            import traceback
            with open('/mnt/data/worker_exc.log','a',encoding='utf-8') as _wf:
                _wf.write('--- worker exception ---\n')
                _wf.write(traceback.format_exc())
                _wf.write('\n')
        except Exception:
            pass
        raise

def repair_worker(args):
    """
    입력: (ruin_port, current_allocations, heuristic_func_name, worker_param_overrides)
    출력: (partial_cost, new_solution_part)
    """
    (ruin_port, current_allocations, heuristic_func_name, worker_params) = args
    G, K, P, F, shortest_paths, passageway_nodes, base_cfg, nei1, nei2, is_pass, gate_parent = _WORKER_CTX

    heuristic_map = {
        'farthest_first': loading_heuristic_farthest_first,
        'flexible_score': loading_heuristic_flexible_score,
    }
    selected_heuristic = heuristic_map[heuristic_func_name]

    # The solve_from function from 쌀먹.py doesn't accept a 'cfg' argument.
    new_solution_part = solve_from(ruin_port, current_allocations, selected_heuristic, P, F, K, shortest_paths, G, passageway_nodes, is_exploring=True)
    if new_solution_part is None:
        return (float('inf'), None)

    # Partial cost (only ports >= ruin_port) — 메인에서 전체비용 재계산하므로 참고용
    partial_cost = calculate_total_cost(new_solution_part, P, F)
    return (partial_cost, new_solution_part)

# ===================================
# 11) Utility: alloc snapshots & more
# ===================================
def _build_alloc_snapshots(best_solution, N, P):
    """ruin 시작 포트별 alloc 스냅샷 미리 만들어서 O(1)로 사용."""
    snaps = [np.full(N, -1, dtype=int)]
    cur = snaps[0].copy()
    for p in range(P-1):  # 마지막 포트 전까지만
        for route, k in best_solution.get(p, []):
            s, e = route[0], route[-1]
            if s == 0:
                cur[e] = k
            elif e == 0:
                if cur[s] == k: cur[s] = -1
            else:
                if cur[s] == k:
                    cur[s] = -1
                    cur[e] = k
        snaps.append(cur.copy())
    return snaps  # len == P

def _precompute_neighbors_and_gate(G, passageway_nodes, N):
    """NEI1/NEI2/IS_PASS/GATE_PARENT"""
    nei1 = [tuple(G[n]) for n in range(N)]
    nei2 = []
    for n in range(N):
        s = set()
        for v in nei1[n]:
            for w in G[v]:
                if w != n and w not in nei1[n]:
                    s.add(w)
        nei2.append(tuple(s))
    is_pass = np.zeros(N, dtype=bool)
    for x in passageway_nodes:
        if 0 <= x < N:
            is_pass[x] = True
    from collections import deque
    parent = {0: None}
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for v in G[u]:
            if v not in parent:
                parent[v] = u
                dq.append(v)
    return nei1, nei2, is_pass, parent


# ============================
# 13) Passageway detection
# ============================
def build_passageway_nodes(G, prob_info):
    try:
        dist_from_gate = nx.single_source_dijkstra_path_length(G, 0, weight='weight')
    except Exception:
        dist_from_gate = nx.single_source_shortest_path_length(G, 0)

    passageway_nodes = set()
    id_to_meta = {}
    id_to_coord = {}
    if 'grid_graph' in prob_info and 'nodes' in prob_info['grid_graph']:
        for coord, meta in prob_info['grid_graph']['nodes']:
            id_to_meta[meta.get('id')] = meta
            try: id_to_coord[meta.get('id')] = tuple(coord)
            except Exception: id_to_coord[meta.get('id')] = (0,0,0)
    def _z_of(n):
        meta = id_to_coord.get(n, None)
        return meta[2] if meta else 0
    z_levels = sorted({_z_of(n) for n in G.nodes()})
    nodes_by_z = {z: [n for n in G.nodes() if _z_of(n)==z] for z in z_levels}
    def _is_ramp(n): return id_to_meta.get(n, {}).get('type') == 'ramp'
    ramps_by_z = {z: [n for n in nodes_by_z[z] if _is_ramp(n)] for z in z_levels}
    ramps_from_below = {z: [] for z in z_levels}
    ramps_to_above   = {z: [] for z in z_levels}
    for z in z_levels:
        for r in ramps_by_z[z]:
            for nb in G.neighbors(r):
                if _z_of(nb) < z:
                    ramps_from_below[z].append(r); break
        for r in ramps_by_z[z]:
            for nb in G.neighbors(r):
                if _z_of(nb) > z:
                    ramps_to_above[z].append(r); break
    if 0 in z_levels:
        H0 = G.subgraph(nodes_by_z[0])
        for r in ramps_by_z.get(0, []):
            if 0 in H0 and r in H0:
                try:
                    p0 = nx.shortest_path(H0, source=0, target=r)
                    passageway_nodes.update(p0[1:-1])
                except nx.NetworkXNoPath:
                    pass
    KSP = 3
    ramp_nodes_all = [n for n in G.nodes() if _is_ramp(n)]
    for z in [zz for zz in z_levels if zz>0]:
        Hz = G.subgraph(nodes_by_z[z])
        incoming = list(dict.fromkeys(ramps_from_below.get(z, [])))
        upward   = list(dict.fromkeys(ramps_to_above.get(z, [])))
        Rz = ramps_by_z.get(z, [])
        if len(incoming)==1 and upward:
            src = incoming[0]
            for t in upward:
                if src in Hz and t in Hz:
                    try:
                        pth = nx.shortest_path(Hz, source=src, target=t)
                        passageway_nodes.update(pth[1:-1])
                    except nx.NetworkXNoPath:
                        pass
        for r in Rz:
            cands = [rp for rp in ramp_nodes_all if _z_of(rp) > z]
            if not cands:
                continue
            best=None; best_len=None
            for rp in cands:
                try:
                    dlen = nx.shortest_path_length(G, source=r, target=rp)
                except nx.NetworkXNoPath:
                    continue
                if (best_len is None) or (dlen < best_len):
                    best_len = dlen; best = rp
            if best is None:
                continue
            try:
                for path in itertools.islice(nx.shortest_simple_paths(G, r, best), KSP):
                    passageway_nodes.update(path[1:-1])
            except nx.NetworkXNoPath:
                pass
    for z in z_levels:
        if len(nodes_by_z.get(z, [])) >= 80:
            Hz = G.subgraph(nodes_by_z[z])
            try:
                arts_z = list(nx.articulation_points(Hz))
            except Exception:
                arts_z = []
            if not arts_z:
                continue
            if z == 0:
                src = 0
            else:
                inc = []
                for r in ramps_by_z.get(z, []):
                    if any(nb in nodes_by_z.get(z-1, []) for nb in G.neighbors(r)):
                        inc.append(r)
                src = inc[0] if inc else None
            if src is None or src not in Hz:
                continue
            ap_z = max(arts_z, key=lambda n: dist_from_gate.get(n, 0))
            try:
                pth = nx.shortest_path(Hz, source=src, target=ap_z)
                passageway_nodes.update(pth[1:-1])
            except nx.NetworkXNoPath:
                pass
    if not passageway_nodes:
        try:
            arts = list(nx.articulation_points(G))
        except Exception:
            arts = []
        if arts:
            ap = max(arts, key=lambda n: dist_from_gate.get(n, 0))
            try:
                pth = nx.shortest_path(G, source=0, target=ap)
                passageway_nodes.update(pth[1:-1])
            except nx.NetworkXNoPath:
                pass
    return passageway_nodes

# =========================
# 14) Main algorithm driver
# =========================
def algorithm(prob_info, timelimit=60,
              scoring_overrides=None,
              pool_param_schedules=None,
              enable_sa=True,
              sa_T0_factor=0.1,
              sa_Tmin=1.0,
              ruin_mode="mixed"
              , timebuffer=None):
    global GLOBAL_START, GLOBAL_TIMELIMIT, _CURRENT_PATHFINDER, NEI1, NEI2, IS_PASS, GATE_PARENT, _GLOBAL_G
    GLOBAL_START = time.time()
    GLOBAL_TIMELIMIT = timelimit
    # --- unified time buffer (stable finish) ---
    try:
        _tb = timebuffer
        if (_tb is None) or (isinstance(_tb, str) and _tb.strip()==""):
            _tb = max(10.0, 0.1 * float(timelimit))
        _tb = float(_tb)
    except Exception:
        _tb = max(10.0, 0.1 * float(timelimit))
    GLOBAL_TIMEBUFFER = max(0.0, min(float(timelimit), _tb))
    GLOBAL_SOFT_LIMIT = max(0.0, float(timelimit) - GLOBAL_TIMEBUFFER)
    # Reduce effective limit so all legacy checks respect buffer
    GLOBAL_TIMELIMIT = GLOBAL_SOFT_LIMIT
    def _time_left_total():
        return float(timelimit) - (time.time() - GLOBAL_START)
    def deadline_near():
        return _time_left_total() <= GLOBAL_TIMEBUFFER
    # Initialize scoring config with overrides
    if scoring_overrides:
        set_global_params(scoring_overrides)

    # Problem
    _record_scoring('init')
    globals()['GLOBAL_SCORING_BASE'] = dict(get_global_params())
    N = prob_info['N']; E = prob_info['E']; K = prob_info['K']; P = prob_info['P']; F = prob_info['F']
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(E)
    _GLOBAL_G = G

    # Shortest path container
    max_num_paths = 5
    shortest_paths = LazyShortestPaths(G, 0, max_num_paths)

    # Passageway detection
    passageway_nodes = build_passageway_nodes(G, prob_info)

    # Bind cached path finder
    _CURRENT_PATHFINDER = PathFinder(G)

    # Precomputes
    NEI1, NEI2, IS_PASS, GATE_PARENT = _precompute_neighbors_and_gate(G, passageway_nodes, N)
    
    # --- [START] 쌀먹.py 로직으로 교체된 초기해 생성 부분 ---
    # phase1_loading_heuristic is now a standalone function from 쌀먹.py
    # It is not a factory anymore.
    def phase1_loading_heuristic(p, node_allocations, demands_to_load, Gparam, passageway_nodes_param, Kparam, is_exploring=True):
        def _backtrack_safe(previous_nodes, start, target):
            if previous_nodes is None: return None
            prev = dict(previous_nodes)
            if start not in prev: prev[start] = None
            path = []; cur = target; seen=set()
            while cur is not None and cur not in seen:
                path.append(cur); seen.add(cur); cur = prev.get(cur, None)
            path.reverse()
            if path and path[0]==start and path[-1]==target: return path
            return None

        grid = prob_info.get('grid_graph', {})
        nodes_raw = grid.get('nodes', [])
        id_to_meta = {meta['id']: meta for coord, meta in nodes_raw} if nodes_raw else {}
        id_to_coord = {meta['id']: tuple(coord) for coord, meta in nodes_raw} if nodes_raw else {}
        def node_type(n): return id_to_meta.get(n, {}).get('type')
        def is_hold(n):
            t = node_type(n)
            if t is not None: return (t=='hold')
            return (n != 0) and (n not in passageway_nodes_param)
        def z_of(n): return id_to_coord.get(n, (0,0,0))[2] if id_to_coord else 0
        z_levels = sorted({z_of(n) for n in Gparam.nodes()})
        nodes_by_z = {z: [n for n in Gparam.nodes() if z_of(n)==z] for z in z_levels}
        def ramps_by_z(z): return [n for n in nodes_by_z[z] if node_type(n)=='ramp']

        # Leaf holds per level
        leaf_nodes=set()
        for z in z_levels:
            H=Gparam.subgraph(nodes_by_z[z])
            for n,deg in H.degree():
                if deg==1 and is_hold(n) and n not in passageway_nodes_param: leaf_nodes.add(n)

        # Distances from gate (0)
        from collections import deque as _dq
        dist_gate = {n: None for n in Gparam.nodes()}
        dq=_dq([0]); dist_gate[0]=0
        while dq:
            u=dq.popleft()
            for v in Gparam.neighbors(u):
                if dist_gate[v] is None:
                    dist_gate[v]=dist_gate[u]+1; dq.append(v)

        # Per-level source depth
        def level_sources(z):
            if z==0: return [0]
            src=[]; low=z-1
            if low in z_levels:
                lower=set(ramps_by_z(low))
                for r in ramps_by_z(z):
                    if any(nbr in lower for nbr in Gparam.neighbors(r)):
                        src.append(r)
            return src
        def per_level_source_depth():
            out={}
            for z in z_levels:
                H=Gparam.subgraph(nodes_by_z[z])
                srcs=level_sources(z)
                dist={n: None for n in H.nodes()}
                dq=_dq()
                for s in srcs:
                    if s in H: dist[s]=0; dq.append(s)
                while dq:
                    u=dq.popleft(); du=dist[u]
                    for v in H.neighbors(u):
                        if dist[v] is None: dist[v]=du+1; dq.append(v)
                out.update(dist)
            return out
        source_depth = per_level_source_depth()
        
        # Peak ports & categories
        Pn = prob_info['P']
        total_holds = sum(1 for n in Gparam.nodes() if is_hold(n))
        def onboard_at_arrival(pidx):
            return sum(q for (o,d), q in Kparam if (o < pidx <= d))
        # Structural no-corridor check: if there is no ramp anywhere, use 30% empty rule
        if not any(node_type(n) == 'ramp' for n in Gparam.nodes()):
            def empties_ratio(pidx):
                onboard = onboard_at_arrival(pidx)
                empties = max(0, total_holds - onboard)
                denom = max(1, total_holds)
                return empties / denom
            peak_ports = {pidx for pidx in range(Pn) if empties_ratio(pidx) <= 0.30}
        else:
            eff_hold = total_holds - sum(1 for n in passageway_nodes_param if is_hold(n))
            peak_ports = {pidx for pidx in range(Pn) if onboard_at_arrival(pidx) > eff_hold}
        # small-destination rule
        tot_by_d = {}
        for (o,d),q in Kparam:
            tot_by_d[d] = tot_by_d.get(d,0) + int(q)
        small_dest = {d for d,t in tot_by_d.items() if t < 5}
        def categorize(d):
            if d == Pn-1:
                return 'terminal'
            if (d in peak_ports) and (d in small_dest):
                return 'peak_small'
            if d in peak_ports:
                return 'peak'
            return 'general'
        # Dynamic Phase-1 range: from p=0 up to just before the first peak port
        first_peak_candidates = sorted([pp for pp in range(1, Pn) if pp in peak_ports])
        first_peak_port = first_peak_candidates[0] if first_peak_candidates else 2  # default to 2 if no peak ports
        if p >= first_peak_port:
            return loading_heuristic_flexible_score(p, node_allocations, demands_to_load, Gparam, passageway_nodes_param, Kparam, is_exploring)

        # Candidates (peak uses penalty 0.9*dist_gate)
        allowed=[n for n in Gparam.nodes() if is_hold(n) and n not in passageway_nodes_param]
        _dg=lambda n: dist_gate.get(n,10**9)
        _sd=lambda n: (10**9 if (source_depth.get(n, None) is None) else source_depth.get(n))
        term_cand = sorted([n for n in allowed if n in leaf_nodes] + [n for n in allowed if n not in leaf_nodes], key=lambda n: (_sd(n), _dg(n)), reverse=True)
        peak_penalty_cand = sorted(allowed, key=lambda n: (0.9*_dg(n), _sd(n)))
        gen_cand  = sorted(allowed, key=lambda n: _dg(n))
        def cand_for(cat):
            if cat=='terminal': return term_cand
            if cat in ('peak','peak_small'): return peak_penalty_cand
            return gen_cand

        # Phase-1 set filtered by first_peak_port, then reordered within p
        phase1_ks = [k for k,((o,d),r) in enumerate(Kparam) if o <= (first_peak_port-1) and d >= (first_peak_port-1) and r>0]
        def _reorder(ulst):
            term = [u for u in ulst if categorize(u[2])=='terminal'];   term.sort(key=lambda t: t[2])
            peak_small = [u for u in ulst if categorize(u[2])=='peak_small']; peak_small.sort(key=lambda t: t[2])
            peak = [u for u in ulst if categorize(u[2])=='peak'];       peak.sort(key=lambda t: t[2])
            general = [u for u in ulst if categorize(u[2])=='general']; general.sort(key=lambda t: t[2], reverse=True)
            return term + peak + peak_small + general
        units_all = _reorder([(k, Kparam[k][0][0], Kparam[k][0][1]) for k in phase1_ks])
        units_p = [u for u in units_all if u[1]==p]

        reserved=set(); used=set(); assign_map={}; assign_paths={}; reservation_order=[]
        def sp_avoid(target, used_nodes, reserved_nodes):
            nodes=set(Gparam.nodes())-set(used_nodes)-set(reserved_nodes)
            nodes.add(target); nodes.add(0)
            H=Gparam.subgraph(nodes)
            try: return nx.shortest_path(H, source=0, target=target, weight=None)
            except nx.NetworkXNoPath: return None

        # Reservation at current origin p: FULL qty with corridor
        for (k,o,d) in units_p:
            c = categorize(d)
            qty = int(demands_to_load.get(k, 0))
            if qty <= 0:
                continue
            assigned_count = 0
            while assigned_count < qty:
                chosen=None; path=None
                for n in cand_for(c):
                    if n in used or n in reserved:
                        continue
                    pth=sp_avoid(n, used, reserved)
                    if pth is not None:
                        chosen=n; path=pth; break
                if chosen is None:
                    break
                used.add(chosen)
                for v in path[1:-1]:
                    reserved.add(v)
                assign_map.setdefault(k, []).append(chosen)
                assign_paths[k]=path
                if k not in reservation_order:
                    reservation_order.append(k)
                assigned_count += 1

        # Peak-only swap
        def _depth(n): return _dg(n)
        peak_assigned=[(k, Kparam[k][0][1], assign_map[k][0]) for k in assign_map.keys() 
                        if categorize(Kparam[k][0][1])=='peak' and assign_map.get(k)]
        improved=True; attempts=0; MAX_ATTEMPTS=200
        while improved and attempts < MAX_ATTEMPTS:
            improved=False; attempts+=1
            peak_assigned.sort(key=lambda t: t[1])
            for idx in range(len(peak_assigned)-1):
                k1,d1,n1 = peak_assigned[idx]
                for jdx in range(idx+1, len(peak_assigned)):
                    k2,d2,n2 = peak_assigned[jdx]
                    if not (d1 < d2): continue
                    if _depth(n2) >= _depth(n1): continue
                    old_p1 = assign_paths.get(k1, None); old_p2 = assign_paths.get(k2, None)
                    used_tmp = set(used); used_tmp.discard(n1); used_tmp.discard(n2)
                    reserved_tmp = set(reserved)
                    if old_p1:
                        for v in old_p1[1:-1]: reserved_tmp.discard(v)
                    if old_p2:
                        for v in old_p2[1:-1]: reserved_tmp.discard(v)
                    p1_new = sp_avoid(n2, used_tmp, reserved_tmp)
                    if p1_new is None: continue
                    reserved_tmp2 = set(reserved_tmp)
                    for v in p1_new[1:-1]: reserved_tmp2.add(v)
                    p2_new = sp_avoid(n1, used_tmp, reserved_tmp2)
                    if p2_new is None: continue
                    assign_map[k1] = [n2]; assign_map[k2] = [n1]
                    assign_paths[k1]=p1_new; assign_paths[k2]=p2_new
                    if old_p1:
                        for v in old_p1[1:-1]: reserved.discard(v)
                    if old_p2:
                        for v in old_p2[1:-1]: reserved.discard(v)
                    for v in p1_new[1:-1]: reserved.add(v)
                    for v in p2_new[1:-1]: reserved.add(v)
                    peak_assigned[idx] = (k1,d1,n2); peak_assigned[jdx] = (k2,d2,n1)
                    improved=True
                    break
                if improved: break

        # Verification
        _peaks = [(k, Kparam[k][0][1], assign_map[k][0]) for k in assign_map.keys() if categorize(Kparam[k][0][1])=='peak' and assign_map.get(k)]
        _peaks.sort(key=lambda t: t[1])
        _ok=True
        for idx in range(len(_peaks)-1):
            _, d1, n1 = _peaks[idx]
            _, d2, n2 = _peaks[idx+1]
            if d1 < d2 and _dg(n2) < _dg(n1): _ok=False; break
        print(f"[Phase-1 verify] peak depth monotone vs dest asc: {'OK' if _ok else 'VIOLATION'}; count={len(_peaks)}")

        # Load order: reservation first, then residuals by priority
        def _cat_of_k(k):  return categorize(Kparam[k][0][1])
        def _dest_of_k(k): return Kparam[k][0][1]
        def _prio_key(k):
            c = _cat_of_k(k)
            pri = 0 if c=='terminal' else 1 if c=='peak' else 2 if c=='peak_small' else 3
            sec = -_dest_of_k(k) if c=='general' else _dest_of_k(k)
            return (pri, sec)
        assigned_first = [k for k in reservation_order if demands_to_load.get(k,0)>0]
        residual_rest  = [k for k,q in demands_to_load.items() if q>0 and k not in reservation_order]
        residual_rest.sort(key=_prio_key)
        # Topo-safe order for Phase-1 (dest <= first_peak_port-1): deeper paths first
        assigned_phase1 = [k for k in assigned_first if Kparam[k][0][1] <= (first_peak_port-1)]
        assigned_phase1.sort(key=lambda k: (len(assign_paths.get(k, [])) if assign_paths.get(k) else 0), reverse=True)
        assigned_other = [k for k in assigned_first if k not in assigned_phase1]
        assigned_first = assigned_phase1 + assigned_other
        load_order = assigned_first + residual_rest

        routes=[]; alloc=node_allocations.copy()
        # Flatten reservations into (k_idx, tgt) pairs limited by current qty
        flat_pairs = []
        for k_idx in load_order:
            qty = int(demands_to_load.get(k_idx, 0))
            if qty <= 0:
                continue
            assigned = list(assign_map.get(k_idx, []))
            take = min(qty, len(assigned))
            for j in range(take):
                tgt = assigned[j]
                flat_pairs.append((k_idx, tgt))
        # Order by farthest from gate first; tie-break by higher z-level first
        def _z(n):
            return id_to_coord.get(n, (0,0,0))[2] if id_to_coord else 0
        flat_pairs.sort(key=lambda kt: (dist_gate.get(kt[1], 0), _z(kt[1])), reverse=True)
        # Commit in that order; keep reservations intact
        for (k_idx, tgt) in flat_pairs:
            if demands_to_load.get(k_idx, 0) <= 0:
                continue
            if alloc[tgt] != -1:
                continue
            dist, prev = util.dijkstra(Gparam, alloc)
            if dist.get(tgt, float('inf')) == float('inf'):
                continue
            path = _backtrack_safe(prev, 0, tgt)
            if path is None:
                continue
            ok = True
            for i in path[:-1]:
                if alloc[i] != -1:
                    ok = False; break
            if not ok:
                continue
            alloc[tgt] = k_idx
            routes.append((path, k_idx))
            demands_to_load[k_idx] = demands_to_load.get(k_idx, 0) - 1

        residual={k:q for k,q in demands_to_load.items() if q>0}
        if residual:
            r2, alloc2 = loading_heuristic_flexible_score(p, alloc, residual, Gparam, passageway_nodes_param, Kparam, is_exploring)
            if r2 is None: return None, None
            routes.extend(r2); alloc=alloc2
        return routes, alloc

    print("Constructing initial solution...")
    initial_allocations = np.ones(N, dtype=int) * -1
    
    def _try_heuristics_sequence():
        heuristics = [
            ("phase1", phase1_loading_heuristic),
            ("flexible_score", loading_heuristic_flexible_score),
            ("farthest_first", loading_heuristic_farthest_first),
        ]
        for tag, h in heuristics:
            try:
                sol = solve_from(0, initial_allocations, h, P, F, K, shortest_paths, G, passageway_nodes, is_exploring=False)
                if sol is not None:
                    print(f"[initial-seed] chosen: {tag}")
                    return sol
                else:
                    if tag == "phase1" and USE_BFS_FRONTIER_FOR_UNLOAD:
                        print("[initial-seed] phase1 returned None with BFS-frontier; retrying with USE_BFS_FRONTIER_FOR_UNLOAD=False...")
                        _backup = USE_BFS_FRONTIER_FOR_UNLOAD
                        try:
                            globals()['USE_BFS_FRONTIER_FOR_UNLOAD'] = False
                            sol2 = solve_from(0, initial_allocations, h, P, F, K, shortest_paths, G, passageway_nodes, is_exploring=False)
                            if sol2 is not None:
                                print("[initial-seed] chosen: phase1 (retry without BFS-frontier)")
                                return sol2
                            else:
                                print("[initial-seed] phase1 retry returned None; trying next...")
                        finally:
                            globals()['USE_BFS_FRONTIER_FOR_UNLOAD'] = _backup
                    else:
                        print(f"[initial-seed] {tag} returned None; trying next...")
            except Exception as e:
                print(f"[initial-seed] {tag} raised: {e}; trying next...")
        return None

    solution = _try_heuristics_sequence()
    if solution is None:
        raise Exception("Failed to construct an initial feasible solution.")
    best_solution = solution
    best_cost = calculate_total_cost(best_solution, P, F)
    print(f"Initial solution cost: {best_cost}")
    # --- [END] 초기해 생성 로직 교체 완료 ---


    # SA schedule (time-based temperature)
    import math, random
    if enable_sa:
        T0 = max(1.0, sa_T0_factor * max(1, best_cost))
        Tmin = max(1e-6, sa_Tmin)
        def current_T():
            progress = min(1.0, max(0.0, (time.time() - GLOBAL_START) / max(1e-6, (timelimit - time_limit_buffer))))
            # Exponential schedule from T0 to Tmin over time
            return T0 * ((Tmin / T0) ** progress)
    else:
        def current_T(): return 1.0

    # Worker pool
    import multiprocessing, os
    num_workers = 4
    # The LNS repair worker will now use the heuristics from 쌀먹.py
    # We remove 'lookahead' from the choices
    # Use a picklable namespace for Windows multiprocessing
    cfg = SimpleNamespace(**get_global_params())
    ctx_payload = (G, K, P, F, shortest_paths, passageway_nodes, cfg, NEI1, NEI2, IS_PASS, GATE_PARENT)
    try:
        if os.environ.get('FORCE_SERIAL_REPAIR','0') != '1':
            pool = multiprocessing.Pool(processes=num_workers, initializer=_init_pool_worker, initargs=(ctx_payload,))
        else:
            pool = None  # serial fallback
            _init_pool_worker(ctx_payload)  # 직렬 모드에서도 컨텍스트 주입
    except Exception as _pool_e:
        print(f"[MP] Pool init failed ({_pool_e}). Fallback to serial.")
        pool = None
        _init_pool_worker(ctx_payload)
    

    iteration_count = 0

    try:
        while time.time() - GLOBAL_START < timelimit - time_limit_buffer:
            iteration_count += 1
            if P <= 1:
                break

            # Alloc snapshots
            alloc_snaps = _build_alloc_snapshots(best_solution, N, P)

            # Build tasks
            tasks = []
            Tnow = current_T()
            alpha = 0.0
            if enable_sa:
                # 고온일수록 큰 ruin 확률↑
                # alpha ∈ [0,1], Tnow가 T0에 가까울수록 1에 근접
                # 여기서 T0는 current_T(0)로 근사
                T0_approx = max(1.0, Tnow)  # 보호용
                alpha = min(1.0, max(0.0, Tnow / T0_approx))

            for w in range(num_workers):
                if _time_left() <= SAFETY_BUFFER_SEC:
                    break

                # Ruin strategy
                if ruin_mode == "single":
                    ruin_start = random.randint(0, P-2)
                else:  # mixed + temperature-aware
                    mode_pick = random.random()
                    base = random.randint(0, P-2)
                    if mode_pick < (0.6 * alpha + 0.2):
                        ruin_start = max(0, base - int(3 * (GLOBAL_CFG.dist2_weight + alpha)))  # 더 앞에서 시작 → 더 큰 rebuild
                    else:
                        ruin_start = base

                current_allocations = alloc_snaps[ruin_start].copy()
                heuristic_choices = ['flexible_score', 'farthest_first']
                selected_heuristic_name = random.choice(heuristic_choices)
                # time-guard: avoid launching a new iteration when within time buffer
                if deadline_near():
                    break
                # worker_params are not used by 쌀먹.py's heuristics.
                tasks.append((ruin_start, current_allocations, selected_heuristic_name, {}))

            # Run workers
            results = []
            if pool is None:
                results = [repair_worker_safe(t) for t in tasks]
            else:
                try:
                    results = pool.map(repair_worker_safe, tasks)
                except Exception as e:
                    import traceback
                    print(f"Multiprocessing pool error: {e}")
                    try:
                        with open('/mnt/data/pool_parent_exc.log','a',encoding='utf-8') as _pf:
                            _pf.write('--- parent exception ---\n')
                            _pf.write(traceback.format_exc())
                            _pf.write('\n')
                    except Exception:
                        pass
                    results = [repair_worker_safe(t) for t in tasks]

            # Evaluate: 단일 SA 수용 (한 후보만 평가/수용)
            candidates = []
            for idx, (_partial_cost, new_solution_part) in enumerate(results):
                if new_solution_part is None:
                    continue
                # 마지막 포트까지 커버하지 못하면 skip
                if max(new_solution_part.keys(), default=-1) != P - 1:
                    continue
                ruin_start = tasks[idx][0]
                merged = dict(best_solution)
                merged.update(new_solution_part)
                total_cost = calculate_total_cost(merged, P, F)
                candidates.append((total_cost, merged))

            if not candidates:
                continue

            # 가장 좋은 후보 하나만 SA 판정
            candidates.sort(key=lambda x: x[0])
            total_cost, merged = candidates[0]
            Delta = total_cost - best_cost
            T = current_T()
            accept = (Delta < 0) or (enable_sa and (np.exp(-Delta / max(1e-6, T)) > np.random.random()))
            if accept:
                best_solution = merged
                best_cost = total_cost
                print(f"[LNS] Iter {iteration_count}: Accepted! Cost: {best_cost} (T={T:.4g})")
                try:
                    globals()['BEST_SCORING_AT_BEST'] = dict(get_global_params())
                    globals()['BEST_SCORING_ITER'] = iteration_count
                except Exception:
                    pass


    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nALNS finished after {iteration_count} iterations.")
    # Report final scoring parameters used when best solution was accepted
    try:
        if BEST_SCORING_AT_BEST is not None:
            print("[RESULT] Scoring params at best solution (iteration %s):" % (str(BEST_SCORING_ITER)))
            for k,v in BEST_SCORING_AT_BEST.items():
                print(f"  - {k} = {v}")
    except Exception as _rep_e:
        print(f"[RESULT] Scoring params report failed: {_rep_e}")
    print(f"Final best cost: {best_cost}")
    return best_solution

# ===========================
# 15) __main__ (IO)
# ===========================
if __name__ == "__main__":
    # You can run this file to test your algorithm from terminal.

    import json
    import os
    import sys
    import jsbeautifier


    def numpy_to_python(obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)  
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        raise TypeError(f"Type {type(obj)} not serializable")
    
    
    # Arguments list should be problem_name, problem_file, timelimit (in seconds)
    if len(sys.argv) == 4:
        prob_name = sys.argv[1]
        prob_file = sys.argv[2]
        timelimit = int(sys.argv[3])

        with open(prob_file, 'r', encoding='utf-8') as f:
            prob_info = json.load(f)

        exception = None
        solution = None

        try:

            alg_start_time = time.time()

            # Run algorithm!
            solution = algorithm(prob_info, timelimit)

            alg_end_time = time.time()


            checked_solution = util.check_feasibility(prob_info, solution)

            checked_solution['time'] = alg_end_time - alg_start_time
            checked_solution['timelimit_exception'] = (alg_end_time - alg_start_time) > timelimit + 2 # allowing additional 2 second!
            checked_solution['exception'] = exception

            checked_solution['prob_name'] = prob_name
            checked_solution['prob_file'] = prob_file


            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
                print(f'Results are saved as file results.json')
                
            sys.exit(0)

        except Exception as e:
            print(f"Exception: {repr(e)}")
            sys.exit(1)

    else:
        print("Usage: python myalgorithm.py <problem_name> <problem_file> <timelimit_in_seconds>")
        sys.exit(2)