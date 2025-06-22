
# =====<{ Imports }>=====

from __future__ import annotations

import math
import random
import heapq
import io
import base64
import networkx as nx

from typing import Dict, List, Set, Tuple, Optional, Any
from itertools import combinations
from typing import NamedTuple

from networkx.algorithms.simple_paths import shortest_simple_paths
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.qasm3 import dumps as circuit_to_qasm3

from flask import Flask, request, jsonify
from flask_cors import CORS

# =====<{ Constants / Network Parameters }>=====

ALPHA = 0.007
BETA  = 0.003
GAMMA = 1.3

SWAP_SUCCESS_PROB = 0.5
SWAP_EFFICIENCY   = 0.99
C_FIBRE_KM_S      = 2.0e5 # km s⁻¹

BASE_GRAPH = nx.Graph()

# =====<{ Math helpers }>=====

def p_func(distance_km: float) -> float:
    return math.exp(-ALPHA * distance_km ** GAMMA)

def f_func(distance_km: float) -> float:
    return math.exp(-BETA * distance_km ** GAMMA)

def F_cost(x: float) -> float:
    return -math.log(x)

def round_trip_time(distance_km: float) -> float:
    return 2.0 * distance_km / C_FIBRE_KM_S

# =====<{ Graph-level helpers }>=====

# ─── Custom Dijkstra from repeater to all nodes ────────────────────────────────────────────────
def repeater_dijkstra(
    r: str
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Single-source Dijkstra from repeater r, where each path's priority is
    cost = F_cost(p_func(D)*f_func(D)),
    but accumulates D = sum of physical distances along the way.
    """
    INF = float("inf")
    best_dist: Dict[str, float] = {r: 0.0}
    best_cost: Dict[str, float] = {r: F_cost(1.0)}  
    parent:    Dict[str, str]  = {}

    # priority queue by cost = F_cost(p(D)*f(D))
    pq: List[Tuple[float, str]] = [(best_cost[r], r)]

    while pq:
        cost_u, u = heapq.heappop(pq)
        if cost_u > best_cost[u]:
            continue

        for v, data in BASE_GRAPH[u].items():
            d = data["cost"]
            D_cand = best_dist[u] + d
            cost_cand = F_cost(p_func(D_cand) * f_func(D_cand))

            if cost_cand < best_cost.get(v, INF):
                best_dist[v] = D_cand
                best_cost[v] = cost_cand
                parent[v]    = u
                heapq.heappush(pq, (cost_cand, v))

    # reconstruct paths
    path_phys: Dict[str, List[str]] = {}
    for u in best_dist:
        seq: List[str] = []
        cur = u
        while True:
            seq.append(cur)
            if cur == r:
                break
            cur = parent[cur]
        seq.reverse()
        path_phys[u] = seq

    return best_dist, path_phys

# ─── Custom Dijkstra for adjacency list graph ────────────────────────────────────────────────
def dijkstra_adj(adj: Dict[State, List[Tuple[State,float]]],
                 start: State,
                 goals: List[State]
) -> Tuple[Optional[float], List[State]]:
    INF = float("inf")
    dist: Dict[State, float] = {start: 0.0}
    prev: Dict[State, State] = {}
    pq = [(0.0, start)]
    goal_set = set(goals)

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue
        if u in goal_set:
            # reconstruct path
            path = [u]
            while u != start:
                u = prev[u]
                path.append(u)
            return d_u, list(reversed(path))
        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None, []

# ─── Build full list of simple and multiple-hop edges ──────────────────────────────────────────
def build_augmented_edges(
    G:       nx.Graph,
    p_th:    float,
    f_th:    float
) -> List[Tuple[str,str,float,List[str]]]:
    """
    Builds all the edges in the augmented graph, which includes:
    - simple edges with cost = F_cost(p_func(d)*f_func(d))
    - virtual edges with cost = F_cost(p_func(D)*f_func(D))
    
    Returns E_aug list of (u, v, cost, path) tuples for all edges.
    """
    E: List[Tuple[str,str,float,List[str]]] = []

    # simple single‐hop edges
    for u, v, data in G.edges(data=True):
        d = data["cost"]
        val = p_func(d) * f_func(d)
        E.append((u, v, F_cost(val), [u, v]))

    # virtual edges via each repeater under cost thresholds
    repeaters = [n for n,d in G.nodes(data=True) if d.get("type")=="repeater"]
    for r in repeaters:
        dist_phys, path_phys = repeater_dijkstra(r)
        for u, D in dist_phys.items():
            if u == r:
                continue
            p_tot = p_func(D)
            f_tot = f_func(D)
            if p_tot <= p_th or f_tot <= f_th:
                continue
            cost = F_cost(p_tot * f_tot)
            E.append((r, u, cost, path_phys[u]))

    return E

# ─── Build state‐graph and find best route ───────────────────────────────────────
State = Tuple[str, str]      
Edge   = Tuple[str,str,float] 

def shortest_cost_with_spdc(
    E_aug:    List[Edge],
    C_swap:   float,
    src:        str,
    tar:        str
) -> Tuple[Optional[float], List[Tuple[str,str]]]:
    """
    Builds the state-space graph for the given augmented edges E_aug, 
    where each edge (u,v,cost) aside from the first (INIT) can be either
    - SPDC (cost 0, sigma='SPDC') meaning it is be used to create entanglement in two directions i.e. adjacent edges
    - SWAP (cost C_swap, sigma='SWAP') meaning it is be used to swap entanglement
    
    Returns best_cost = minimum cost to reach (tar, *); state_path = list of states (node, sigma) from (src, 'INIT') to (tar, *)
    """

    # initialize adjacency for all states
    states = [(x, sigma) for x in {u for u,_,_ in E_aug} | {v for _,v,_ in E_aug}
                     for sigma in ['INIT','SPDC','SWAP']]
    adj: Dict[State, List[Tuple[State,float]]] = {st: [] for st in states}

    # helper: add a transition u_st -> v_st with cost c
    def add_tr(u_st: State, v_st: State, c: float):
        adj[u_st].append((v_st, c))

    # helper: attempt to add a transition from u_st -> some v,sigma2
    def try_add(sigma2: str, pen: float):
        if v == tar:
            # if v is sink, we keep sigma2 same as previous or INIT
            add_tr(u_st, (tar, sigma), c_e + pen)
        else:
            add_tr(u_st, (v, sigma2), c_e + pen)
            
    # populate transitions
    for u, v, c_e in E_aug:
        for sigma in ['INIT','SPDC','SWAP']:
            u_st = (u, sigma)

            if sigma == 'INIT':
                # first edge: can either SPDC or swap
                try_add('SPDC', 0.0)
                try_add('SWAP', C_swap)
            else:
                # middle edges: no SPDC -> SPDC
                if sigma != 'SPDC':
                    try_add('SPDC', 0.0)
                try_add('SWAP', C_swap)

    # Dijkstra from (src, 'INIT') to any (tar, *)
    goal_states = [(tar, sigma) for sigma in ['INIT','SPDC','SWAP']]
    best_cost, state_path = dijkstra_adj(adj, (src,'INIT'), goal_states)
    if best_cost is None:
        return None, []

    return best_cost, state_path

# ─── Builds all the necessary prerequisites for the entanglement routing path ────────────────────────────────
def find_entanglement_path(
    graph: nx.Graph,
    src: str,
    dst: str,
    swap_penalty: float,
    p_threshold: float,
    f_threshold: float
) -> Tuple[
    List[str],            # logical node chain
    List[Tuple[str,str]], # state_path (node,phase)
    List[List[str]],      # phys_paths
    List[float]           # hop costs
]:
    """
    Find the best path from src to dst in the state space graph,
    using the augmented edges built from the physical graph.
    
    Returns node_path = list of logical nodes in the path, state_path = list of (node, phase) tuples for each hop,
        phys_paths = list of physical paths corresponding to each hop, costs = list of costs for each hop in the path
    """
    
    E_aug = build_augmented_edges(graph, p_threshold, f_threshold)

    E_edges: List[Tuple[str,str,float]] = []
    for u, v, cost, _ in E_aug:
        E_edges.append((u, v, cost))
        E_edges.append((v, u, cost))

    best_cost, state_path = shortest_cost_with_spdc(
        E_edges, swap_penalty, src, dst
    )
    
    if best_cost is None:
        return [], [], [], []

    node_path:    List[str]       = [st[0] for st in state_path]
    phys_paths:   List[List[str]] = []
    costs:        List[float]     = []

    # for each hop in the state_path, find the matching entry in E_aug
    for (u,_), (v,_) in zip(state_path, state_path[1:]):
        for uu, vv, c, phys in E_aug:
            if {uu, vv} == {u, v}:
                phys_paths.append(phys)
                costs.append(c)
                break

    return node_path, state_path, phys_paths, costs

# ─── Build the work graph for the simulations ─────────────────────────────────────
def build_work_graph(
    base: nx.Graph,
    node_path:   List[str],
    phys_paths:  List[List[str]],
    state_path:  List[Tuple[str,str]],
    edge_keys:   List[str]
) -> nx.MultiGraph:
    """
    Build the work graph Gw from the logical node path, physical paths,
    state path, and edge keys.
    The built graph has super edges that are made of individual single or virtual
    edges clustered around their generators.
        
    Returns the constructed MultiGraph with the above properties.
    """
    
    Gw = nx.MultiGraph()
    i = 0
    
    while i < len(node_path) - 1:
        u = node_path[i]
        
        raw1 = phys_paths[i]
        if raw1[0] != u:
            # path is backwards, flip it
            raw1 = list(reversed(raw1))

        if (i + 2 < len(node_path) and state_path[i+1][1] == "SPDC"):
            rep  = node_path[i+1]
            raw2 = phys_paths[i+1]
            if raw2[0] != rep:
                raw2 = list(reversed(raw2))

            p1, p2 = raw1, raw2

            # total distances per sub‐chain
            D1 = sum(base[a][b]["cost"] for a,b in zip(p1, p1[1:]))
            D2 = sum(base[a][b]["cost"] for a,b in zip(p2, p2[1:]))

            p_tot = p_func(D1) * p_func(D2)
            f_tot = f_func(D1) * f_func(D2)
            delay = max(round_trip_time(D1), round_trip_time(D2))
            key = f"{edge_keys[i]}|{edge_keys[i+1]}"
            
            Gw.add_edge(u, node_path[i+2],
                key=key, mode="spdc",
                p_succ=p_tot, fidelity=f_tot,
                length=D1+D2, 
                delay=delay)
            
            i += 2 
        else:
            v = node_path[i+1]
            D = sum(
                base[a][b]["cost"]
                for a, b in zip(raw1, raw1[1:])
            )
            p_tot = p_func(D)
            f_tot = f_func(D)
            delay = round_trip_time(D)
            key  = edge_keys[i]
            mode = "spdc" if state_path[i+1][1]=="SPDC" else "fibre"
            
            Gw.add_edge(u, v,
                        key=key, mode=mode,
                        p_succ=p_tot, fidelity=f_tot,
                        length=D, delay=delay)

            i += 1

    return Gw

# ─── Build the full physical chain from logical nodes and physical paths ───────────────
def build_full_physical_chain(
    logical_nodes: List[str],
    phys_paths:    List[List[str]]
) -> List[str]:
    full: List[str] = []

    for i, P in enumerate(phys_paths):
        if P[0] != logical_nodes[i]:
            P = list(reversed(P))

        if i == 0:
            full.extend(P)
        else:
            full.extend(P[1:])

    return full

# ─── Collapse SPDC hops into single links (L - repeater - R) ───────────────────────
def collapse_with_phys(
    node_path:   List[str],
    state_path:  List[Tuple[str,str]],
    edge_keys:   List[str],
    phys_paths:  List[List[str]]
) -> Tuple[
    List[str],             # collapsed_nodes
    List[str],             # collapsed_edge_keys
    List[List[str]],       # collapsed_phys_paths
    List[Tuple[str,str]]   # collapsed_state
]:
    new_nodes:       List[str]            = []
    new_edge_keys:   List[str]            = []
    new_phys_paths:  List[List[str]]      = []
    new_state:       List[Tuple[str,str]] = []

    """
    Collapse any two hops around a SPDC repeater into one link.
    For example, if the path is:
        N0 -- E0 -- SPDC -- E1 -- N2
    where E0 and E1 are physical paths, this will collapse it to:
        N0 -- (E0|E1) -- N2
    where (E0|E1) is the combined physical path from N0 to N2 via the repeater.
    
    Returns:
        collapsed_nodes:        List of logical nodes in the collapsed path
        collapsed_edge_keys:    List of edge keys for the collapsed path
        collapsed_phys_paths:   List of physical paths for the collapsed path
        collapsed_state:        List of (node, phase) tuples for the collapsed path
    """

    i = 0
    while i < len(edge_keys):
        if (i+1 < len(edge_keys)
            and state_path[i+1][1] == "SPDC"
        ):
            u = node_path[i]
            v = node_path[i+2]
            k1, k2 = edge_keys[i], edge_keys[i+1]
            
            p1 = phys_paths[i][:]
            if p1[0] != u: p1.reverse()
            p2 = phys_paths[i+1][:]
            if p2[0] != node_path[i+1]: p2.reverse()
            comp_p = p1 + p2[1:]

            new_nodes.append(u)
            new_edge_keys.append(f"{k1}|{k2}")
            new_phys_paths.append(comp_p)
            new_state.append(state_path[i])

            i += 2
        else:
            u = node_path[i]
            k = edge_keys[i]
            p = phys_paths[i][:]
            if p[0] != u: p.reverse()

            new_nodes.append(u)
            new_edge_keys.append(k)
            new_phys_paths.append(p)
            new_state.append(state_path[i])

            i += 1

    new_nodes.append(node_path[-1])
    new_state.append(state_path[-1])

    return new_nodes, new_edge_keys, new_phys_paths, new_state

# ─── Check if an edge respects triangle inequalities ───────────────────────────────
def edge_respects_triangle(a: str, b: str, cost: float) -> bool:
    """
    Allow the edge (a, b)=cost only if, for every node c that is directly
    connected to both a and b, the triangle inequalities hold:
        cost      ≤ w(a,c) + w(c,b)
        w(a,c) ≤ cost    + w(c,b)
        w(c,b) ≤ cost    + w(a,c)
        
    Returns True if the edge respects all triangle inequalities, False otherwise.
    """
    # Find common neighbors of a and b (nodes with direct edges to both)
    neighbors_a = set(BASE_GRAPH.neighbors(a))
    neighbors_b = set(BASE_GRAPH.neighbors(b))
    common   = neighbors_a & neighbors_b

    for c in common:
        w_ac = BASE_GRAPH[a][c]["cost"]
        w_bc = BASE_GRAPH[b][c]["cost"]
        
        print(f"Checking triangle inequalities for edge ({a}, {b}) with cost {cost} and common neighbor {c}:")
        print(f"  w({a}, {c}) = {w_ac}, w({b}, {c}) = {w_bc}")

        if cost > w_ac + w_bc:
            return False
        if w_ac > cost + w_bc:
            return False
        if w_bc > cost + w_ac:
            return False

    return True


# =====<{ Simulators }>=====

# ─── LinkKey and related helpers ────────────────────────────────────────────────
class LinkKey(NamedTuple):
    u: str
    v: str
    key: str

def make_link_key(u: str, v: str, key: str) -> LinkKey:
    # normalize u/v so ordering doesn't matter
    return LinkKey(*( (u,v,key) if u <= v else (v,u,key) ))

LinkKeyState = Dict[LinkKey, Dict[str,Any]]
Memory    = Dict[str, List[Tuple[str,float]]]
Operation = Dict[str, Any]

# ───── Continuous‐time probabilistic simulator ───────────────────────────────
def simulate_continuous_prob(
    path_nodes: List[str],
    path_edges: List[str],
    network: nx.MultiGraph,
    state_path: List[Tuple[str,str]]
) -> Tuple[Optional[float], Memory, List[Operation]]:
    """
    Continuous-time probabilistic simulator for entanglement routing.
    This simulates the generation of links between nodes in the path,
    and the subsequent swaps at repeaters in divide-et-impera order, 
    using a scheduler via a priority queue until either:
        - a final link is formed between the start and end nodes, or
        - no more links can be formed due to failures.
    
    Returns:
        final_fid:  Final fidelity of the path, or None if no path was formed.
        memory:     Memory of links at each node, mapping node to list of (neighbor, fidelity).
        operations: List of operations performed during the simulation.
    """
    
    memory: Memory = {n: [] for n in path_nodes}
    operations: List[Operation] = []
    link_states: Dict[LinkKey, Dict[str,Any]] = {}

    # Event queue: heap of (time, unique_id, event_type, data)
    event_queue: List[Tuple[float,int,str,Dict[str,Any]]] = []
    
    # a simple counter so no two entries compare equal on (time,uid)
    event_counter = [0]  

    # DEI‐swap heap: (segment_size, repeater_node)
    swap_heap: List[Tuple[int,str]] = []
    seg_size_map: Dict[str,int] = {}
    segment_bounds: Dict[str,Tuple[int,int]] = {}

    # ─── Helper: collect all swap repeaters in DEI order ─────────────────────────────
    def collect_swaps(lo: int, hi: int):
        """
        Walk the interval [lo,hi] in DEI order, and for each midpoint (a swap repeater)
        that needs swapping, record:
            - swap_heap entry (seg_size, repeater)
            - seg_size_map[repeater] = seg_size
            - segment_bounds[repeater] = (lo,hi)
        Then recurse on children.
        """
        if hi - lo < 2:
            return

        mid = (lo + hi)//2
        repeater, phase = state_path[mid]

        if phase == "SWAP":
            seg_size = hi - lo
            swap_heap.append((seg_size, repeater))
            seg_size_map[repeater] = seg_size
            segment_bounds[repeater] = (lo, hi)

        collect_swaps(lo, mid)
        collect_swaps(mid, hi)

    collect_swaps(0, len(path_nodes)-1)
    heapq.heapify(swap_heap)

    # ─── Helper: find two ready links at a repeater ─────────────────────────
    def two_ready(repeater: str) -> Optional[List[Tuple[str,float]]]:
        ready_pairs: List[Tuple[str,float]] = []
        for lk, st in link_states.items():
            if not st["ready"]:
                continue

            if repeater == lk.u:
                peer = lk.v
            elif repeater == lk.v:
                peer = lk.u
            else:
                continue

            ready_pairs.append((peer, st["fidelity"]))
            if len(ready_pairs) >= 2:
                return ready_pairs[:2]
        return None

    # ─── Helper: schedule a link generation event ───────────────────────────────
    def schedule_link(u: str, v: str, key: str, start_t: float):
        if not network.has_edge(u, v, key):
            if network.has_edge(v, u, key):
                u, v = v, u
            else:
                raise KeyError(f"Edge {u}-{v} with key {key} not in work_graph")
        data = network[u][v][key]     
                
        dist = data.get("cost", data.get("length"))
        
        if data.get("mode") == "spdc":
            p_succ = data["p_succ"]
            fid    = data["fidelity"]
            delay  = data["delay"]
        else:
            p_succ = p_func(dist)
            fid    = f_func(dist)
            delay  = round_trip_time(dist)

        lk = make_link_key(u, v, key)
        if lk not in link_states:
            link_states[lk] = {"ready": False, "fidelity": 0.0, "length": dist}

        event_counter[0] += 1
        eid = event_counter[0]
        event_t = start_t + delay + random.random() * 1e-9
        
        heapq.heappush(event_queue, (event_t, eid, "link_complete", {
            "u":u, "v":v, "key":key,
            "p_succ":p_succ, "fidelity":fid, "length": dist,
            "start_t": start_t
        }))
        
    # ─── Helper: try to schedule swaps at repeaters ────────────────────────────────
    def try_schedule_swaps(now: float):
        """
        In DEI order, pop each repeater whose two adjacent links are both ready.
        Uses two_ready() to discover dynamic neighbors.
        """
        while swap_heap:
            seg_size, repeater = swap_heap[0]
            ready_links = two_ready(repeater)
            if not ready_links:
                break

            # we have two ready links, reserve them
            (nbr1, f1), (nbr2, f2) = ready_links
            heapq.heappop(swap_heap)

            # identify the exact LinkKey objects
            lk1 = next(lk for lk in link_states if set((lk.u, lk.v)) == {repeater, nbr1})
            lk2 = next(lk for lk in link_states if set((lk.u, lk.v)) == {repeater, nbr2})
            
            link_states[lk1]["ready"] = False
            link_states[lk2]["ready"] = False

            L1 = link_states[lk1]["length"]
            L2 = link_states[lk2]["length"]

            total_len = L1 + L2
            
            delay1 = round_trip_time(L1)
            delay2 = round_trip_time(L2)
            swap_delay = max(delay1, delay2)
            
            event_counter[0] += 1
            eid = event_counter[0]
            heapq.heappush(event_queue, (
                now + swap_delay, eid, "swap_attempt", {
                    "repeater":   repeater,
                    "lk1":        lk1,
                    "lk2":        lk2,
                    "f1":         f1,
                    "f2":         f2,
                    "total_len":  total_len,
                    "seg_size":   seg_size
                }
            ))
            
    # ─── Helper: retry a link generation if it went down ──────────────────────────────
    def retry_link(u: str, v: str, key: str):
        if network.has_edge(u, v, key):
            # if the link is already in the work graph, just reschedule it
            schedule_link(u, v, key, now)
        else:
            i_u = path_nodes.index(u)
            i_v = path_nodes.index(v)
            step = 1 if i_v > i_u else -1
            # walk from u to v in the collapsed path, scheduling each original constituent edge
            for idx in range(i_u, i_v, step):
                a = path_nodes[idx]
                b = path_nodes[idx+step]
                ek = (path_edges[idx] if step == 1 else path_edges[idx-1])
                schedule_link(a, b, ek, now)

    # Initial link scheduling    
    for u,v,key in zip(path_nodes, path_nodes[1:], path_edges):
        schedule_link(u, v, key, 0.0)

    final_fid: Optional[float] = None

    # Main event loop --- process events in order of time
    while event_queue:
        now, _eid, ev_type, data = heapq.heappop(event_queue)

        # link generation attempt event handler
        if ev_type == "link_complete":
            u, v, key = data["u"], data["v"], data["key"]
            lk = make_link_key(u, v, key)

            # failure -> retry
            if random.random() > data["p_succ"]:
                operations.append({
                    "time": now,
                    "type": "link_generation",
                    "nodes": [u, v],
                    "key":   key,
                    "status":"failed",
                    "retry": True
                })
                schedule_link(u, v, key, now)
                continue

            # success -> record & try swaps
            link_states[lk]["ready"]    = True
            link_states[lk]["fidelity"] = data["fidelity"]
            memory[u].append((v, data["fidelity"]))
            memory[v].append((u, data["fidelity"]))
            operations.append({
                "time": now,
                "type": "link_generation",
                "nodes": [u, v],
                "key":   key,
                "status":"success",
                "fidelity": data["fidelity"]
            })
            try_schedule_swaps(now)

        # swap attempt event handler
        elif ev_type == "swap_attempt":
            rep = data["repeater"]
            lk_left  = data["lk1"]
            lk_right = data["lk2"]
            f_left, f_right = data["f1"], data["f2"]

            # failure -> tear down entire link & retry later
            if random.random() > SWAP_SUCCESS_PROB:
                operations.append({
                    "time": now,
                    "type": "swap",
                    "node": rep,
                    "inputs": [(lk_left.u if lk_left.u!=rep else lk_left.v, f_left),
                               (lk_right.u if lk_right.u!=rep else lk_right.v, f_right)],
                    "status": "failed",
                    "retry": True
                })
                # purge memory
                peers = {lk_left.u,lk_left.v,lk_right.u,lk_right.v}
                memory[rep] = [(n,f) for n,f in memory[rep] if n not in peers]
                
                # rebuild L-R links
                retry_link(lk_left.u, lk_left.v, lk_left.key)
                retry_link(lk_right.u, lk_right.v, lk_right.key)
                
                # requeue the lost swaps over the failed link in the heap 
                lo, hi = segment_bounds[rep]
                for r, (lo2, hi2) in segment_bounds.items():
                    if lo2 >= lo and hi2 <= hi:
                        heapq.heappush(swap_heap, (seg_size_map[r], r))
                continue

            # success -> combine into bypass link
            new_fid = f_left * f_right * SWAP_EFFICIENCY
            operations.append({
                "time": now,
                "type": "swap",
                "node": rep,
                "inputs": [(lk_left.u if lk_left.u!=rep else lk_left.v, f_left),
                           (lk_right.u if lk_right.u!=rep else lk_right.v, f_right)],
                "status": "success",
                "result_fidelity": new_fid
            })

            # purge old and stale entries
            left_peer  = lk_left.u if lk_left.u!=rep else lk_left.v
            right_peer = lk_right.u if lk_right.u!=rep else lk_right.v
            memory[rep] = [(n,f) for n,f in memory[rep] if n not in (left_peer, right_peer)]
            memory[left_peer]  = [(n,f) for n,f in memory[left_peer] if n != rep]
            memory[right_peer] = [(n,f) for n,f in memory[right_peer] if n != rep]
            
            # register the new bypass link
            memory[left_peer].append((right_peer, new_fid))
            memory[right_peer].append((left_peer, new_fid))
            
            bypass_key = f"{lk_left.key}|{lk_right.key}"
            kbypass = make_link_key(left_peer, right_peer, bypass_key)
            
            link_states[kbypass] = {
                "ready":    True,
                "fidelity": new_fid,
                "length":   data["total_len"]
            }
            
            # check to see if end-to-end link was formed
            endpoints = {path_nodes[0], path_nodes[-1]}
            if {left_peer, right_peer} == endpoints:
                final_fid = new_fid
                break

            # try to schedule any further swaps now that a new link exists
            try_schedule_swaps(now)

    return final_fid, memory, operations

# ───── Discrete‐time probabilistic simulator ────────────────────────────────
def simulate_discrete_prob(
    path_nodes: List[str],
    path_edges: List[str],
    network: nx.MultiGraph
) -> Tuple[Optional[float], Dict[str,List[Tuple[str,float]]], List[Dict[str,Any]]]:
    """
    Step-driven: retry link-generation until success, then do DEI swaps
    in increasing granularity, retrying failed swaps with full sub-tree redo.
    
    Returns:
        final_fid:  Final fidelity of the path, or None if no path was formed.
        memory:     Memory of links at each node, mapping node to list of (neighbor, fidelity).
        operations: List of operations performed during the simulation.
    """

    memory: Dict[str,List[Tuple[str,float]]] = {n: [] for n in path_nodes}
    operations: List[Dict[str,Any]] = []

    step = 0
    
    # ─── Helper: next step counter ───────────────────────────────────────────────
    def next_step() -> int:
        nonlocal step
        step += 1
        return step

    # ─── Helper: generate links between the given path indexes ───────────────────────────────────────
    def generate_links(lo: int, hi: int):
        for i in range(lo, hi):
            u, v, key = path_nodes[i], path_nodes[i+1], path_edges[i]
            data = network[u][v][key]

            dist   = data["length"]
            p_succ = data.get("p_succ", p_func(dist))
            fid    = data.get("fidelity", f_func(dist))

            # retry until success
            while True:
                t = next_step()
                if random.random() <= p_succ:
                    operations.append({
                        "time": t,
                        "type": "link_generation",
                        "nodes":[u,v],
                        "key": key,
                        "status":"success",
                        "fidelity": fid
                    })
                    memory[u].append((v,fid))
                    memory[v].append((u,fid))
                    break
                else:
                    operations.append({
                        "time": t,
                        "type": "link_generation",
                        "nodes":[u,v],
                        "key": key,
                        "status":"failed",
                        "retry": True
                    })

    # ─── Helper: perform DEI swaps recursively ─────────────────────────────────────
    def perform_swaps(lo: int, hi: int):
        if hi - lo < 2:
            return
        mid = (lo + hi)//2

        # 1) do all smaller swaps first
        perform_swaps(lo, mid)
        perform_swaps(mid, hi)

        # 2) now attempt the swap at 'rep' if *any* two links are ready
        rep = path_nodes[mid]
        if len(memory[rep]) < 2:
            return

        # pick the first two ready links in memory[rep]
        (nbr1, f1), (nbr2, f2) = memory[rep][0], memory[rep][1]
        t = next_step()

        if random.random() <= SWAP_SUCCESS_PROB:
            # ─── swap success ─────────────────────────
            new_f = f1 * f2 * SWAP_EFFICIENCY
            operations.append({
                "time":            t,
                "type":            "swap",
                "node":            rep,
                "inputs":         [(nbr1, f1), (nbr2, f2)],
                "status":         "success",
                "result_fidelity": new_f
            })
            # purge old entries pointing to this repeater
            memory[nbr1] = [(n,f) for n,f in memory[nbr1] if n != rep]
            memory[nbr2] = [(n,f) for n,f in memory[nbr2] if n != rep]
            memory[rep]  = []
            # add the bypass link
            memory[nbr1].append((nbr2, new_f))
            memory[nbr2].append((nbr1, new_f))

        else:
            # ─── swap failure ─────────────────────────
            operations.append({
                "time":    t,
                "type":    "swap",
                "node":    rep,
                "inputs": [(nbr1, f1), (nbr2, f2)],
                "status":  "failed",
                "retry":   True
            })
            # clear this repeater’s memory
            memory[rep] = []
            # regenerate the entire sub‐path [lo,hi)
            for i in range(lo, hi):
                generate_links(i, i+1)
            # retry this subtree
            perform_swaps(lo, hi)

    # Generate all initial links in the full path
    generate_links(0, len(path_nodes)-1)

    # Start performing swaps over the full path
    perform_swaps(0, len(path_nodes)-1)

    final_fid: Optional[float] = None
    for nbr, f in memory[path_nodes[0]]:
        if nbr == path_nodes[-1]:
            final_fid = f
            break

    return final_fid, memory, operations

# =====<{ Qiskit circuit builder >=====

def build_circuit_v2(path: List[str], ops: List[Operation]) -> Tuple[str, str]: 
    """
    Build a Qiskit circuit from the given path and operations.
    The path is a list of logical nodes, and ops is a list of operations
    performed during the simulation, including link generations and swaps.
    
    Returns:
        - qasm_str: QASM3 string representation of the circuit
        - img_b64:  Base64-encoded PNG image of the circuit diagram
    """
    
    qubit_map: Dict[str, List[int]] = {}
    next_qubit = 0
    for idx, node in enumerate(path):
        if idx == 0 or idx == len(path) - 1:
            qubit_map[node] = [next_qubit]
            next_qubit += 1
        else:
            # [0] for left‐link, [1] for right‐link
            qubit_map[node] = [next_qubit, next_qubit + 1]
            next_qubit += 2

    num_reps   = len(path) - 2
    num_clbits = 2 * num_reps + 2
    
    qreg = QuantumRegister(next_qubit, name="q")
    creg = ClassicalRegister(num_clbits, name="c")

    qc = QuantumCircuit(qreg, creg)

    def rep_clbits(rep_idx: int) -> Tuple[int,int]:
        base = 2 * (rep_idx - 1)
        return base, base + 1

    end_c0 = num_clbits - 2
    end_c1 = num_clbits - 1

    for op in sorted(ops, key=lambda d: d["time"]):
        
        if op["type"] == "link_generation":
            u, v = op["nodes"]
            i_u = path.index(u)
            qu_u = (qubit_map[u][1]
                    if i_u not in (0, len(path)-1)
                    else qubit_map[u][0])
            qu_v = qubit_map[v][0]  
            
            qc.h(qu_u)
            qc.cx(qu_u, qu_v)
            if op["status"] != "success":
                qc.reset(qu_u)
                qc.reset(qu_v)

        elif op["type"] == "swap":
            rep = op["node"]
            r_idx = path.index(rep)
            ql, qr = qubit_map[rep] 
            c0, c1 = rep_clbits(r_idx)

            qc.cx(ql, qr)
            qc.h(ql)
            
            qc.measure(ql, creg[c0])
            qc.measure(qr, creg[c1])
            
            qc.reset(ql)
            qc.reset(qr)

            if op["status"] == "success":
                x, _ = op["inputs"][0]  
                y, _ = op["inputs"][1]  

                xi, yi = path.index(x), path.index(y)
                if len(qubit_map[x]) == 1:
                    qs_x = qubit_map[x][0]
                else:
                    qs_x = qubit_map[x][1] if xi < r_idx else qubit_map[x][0]

                if len(qubit_map[y]) == 1:
                    qs_y = qubit_map[y][0]
                else:
                    qs_y = qubit_map[y][1] if yi < r_idx else qubit_map[y][0]

                # Apply conditional corrections based on the measurement results
                with qc.if_test((creg, 1 << c0)):
                    qc.z(qreg[qs_x])

                with qc.if_test((creg, 1 << c1)):
                    qc.x(qreg[qs_x])

                with qc.if_test((creg, 1 << c0)):
                    qc.z(qreg[qs_y])

                with qc.if_test((creg, 1 << c1)):
                    qc.x(qreg[qs_y])

        else:
            pass

    qc.measure(qubit_map[path[0]][0], creg[end_c0])
    qc.measure(qubit_map[path[-1]][0], creg[end_c1])

    qasm_str = circuit_to_qasm3(qc)

    fig = qc.draw(output="mpl", fold=100)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return qasm_str, img_b64


# =====<{ Flask endpoints }>=====

app = Flask(__name__)
CORS(app)

@app.route("/config", methods=["GET"])
def get_config():
    """Return the current simulation parameters."""
    return jsonify({
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": GAMMA,
        "swap_success_prob": SWAP_SUCCESS_PROB,
        "swap_efficiency": SWAP_EFFICIENCY,
        "c_fibre_km_s": C_FIBRE_KM_S
    }), 200

@app.route("/config", methods=["POST"])
def set_config():
    """
    Update one or more simulation parameters.
    Expects JSON with any of: alpha, beta, gamma,
    swap_success_prob, swap_efficiency, c_fibre_km_s.
    """
    data = request.json or {}
    errs = []

    def set_param(key: str, attr_name: str, validator=lambda v: v > 0):
        """Helper to set a global parameter with basic validation."""
        if key in data:
            val = data[key]
            try:
                fv = float(val)
                if not validator(fv):
                    raise ValueError(f"{key} must be positive")
            except Exception as e:
                errs.append(f"{key}: {e}")
                return
            globals()[attr_name] = fv

    global ALPHA, BETA, GAMMA
    global SWAP_SUCCESS_PROB, SWAP_EFFICIENCY, C_FIBRE_KM_S

    set_param("alpha",               "ALPHA")
    set_param("beta",                "BETA")
    set_param("gamma",               "GAMMA")
    set_param("swap_success_prob",   "SWAP_SUCCESS_PROB",   validator=lambda v: 0 <= v <= 1)
    set_param("swap_efficiency",     "SWAP_EFFICIENCY",     validator=lambda v: 0 <= v <= 1)
    set_param("c_fibre_km_s",        "C_FIBRE_KM_S",        validator=lambda v: v > 0)

    if errs:
        return jsonify({"error": "Invalid parameters", "details": errs}), 400

    # Return the new configuration
    return jsonify({
        "status": "configuration updated",
        "config": {
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "swap_success_prob": SWAP_SUCCESS_PROB,
            "swap_efficiency": SWAP_EFFICIENCY,
            "c_fibre_km_s": C_FIBRE_KM_S
        }
    }), 200

# ---- node management -------------------------------------------------------
@app.route("/add_endpoint", methods=["POST"])
def add_endpoint():
    node_id = request.json["id"]
    if BASE_GRAPH.has_node(node_id):
        return jsonify({"error": "Node already exists"}), 400
    BASE_GRAPH.add_node(node_id, type="endpoint")
    return jsonify({"status": "endpoint added", "id": node_id}), 200

@app.route("/add_repeater", methods=["POST"])
def add_repeater():
    node_id = request.json["id"]
    if BASE_GRAPH.has_node(node_id):
        return jsonify({"error": "Node already exists"}), 400
    BASE_GRAPH.add_node(node_id, type="repeater")
    return jsonify({"status": "repeater added", "id": node_id}), 200

@app.route("/remove_node", methods=["POST"])
def remove_node():
    node_id = request.json["id"]
    if not BASE_GRAPH.has_node(node_id):
        return jsonify({"error": "Node does not exist"}), 400
    BASE_GRAPH.remove_node(node_id)
    return jsonify({"status": "node removed", "id": node_id}), 200

# ---- edge management -------------------------------------------------------
@app.route("/add_edge", methods=["POST"])
def add_edge():
    a, b, cost = request.json["node1"], request.json["node2"], request.json["cost"]
    if not BASE_GRAPH.has_node(a) or not BASE_GRAPH.has_node(b):
        return jsonify({"error": "One or both nodes do not exist"}), 400
    if BASE_GRAPH.has_edge(a, b):
        return jsonify({"error": "Edge already exists"}), 400
    if not edge_respects_triangle(a, b, cost):
        return jsonify({"error": "Triangle inequality violated"}), 400
    BASE_GRAPH.add_edge(a, b, cost=cost)
    return jsonify({"status": "edge added", "nodes": [a, b], "cost": cost}), 200

@app.route("/modify_edge", methods=["POST"])
def modify_edge():
    a, b, cost = request.json["node1"], request.json["node2"], request.json["new_cost"]
    if not BASE_GRAPH.has_edge(a, b):
        return jsonify({"error": "Edge does not exist"}), 400
    if not edge_respects_triangle(a, b, cost):
        return jsonify({"error": "Triangle inequality violated"}), 400
    BASE_GRAPH[a][b]["cost"] = cost
    return jsonify({"status": "edge modified", "nodes": [a, b], "new_cost": cost}), 200

@app.route("/remove_edge", methods=["POST"])
def remove_edge():
    a, b = request.json["node1"], request.json["node2"]
    if not BASE_GRAPH.has_edge(a, b):
        return jsonify({"error": "Edge does not exist"}), 400
    BASE_GRAPH.remove_edge(a, b)
    return jsonify({"status": "edge removed", "nodes": [a, b]}), 200

# ---- network management ----------------------------------------------------
@app.route("/clear_network", methods=["DELETE"])
def clear_network():
    BASE_GRAPH.clear()
    return jsonify({"status": "network cleared"}), 200

@app.route("/import_network", methods=["POST"])
def import_network():
    data = request.json
    BASE_GRAPH.clear()
    for node in data.get("nodes", []):
        BASE_GRAPH.add_node(node["id"], type=node.get("type", "endpoint"))
    for edge in data.get("edges", []):
        a, b = edge["source"]["id"], edge["target"]["id"]
        BASE_GRAPH.add_edge(a, b, cost=edge["value"])
    return jsonify({"status": "network imported",
                    "nodeCount": BASE_GRAPH.number_of_nodes(),
                    "edgeCount": BASE_GRAPH.number_of_edges()}), 200

# ---- entanglement request --------------------------------------------------
@app.route("/request_entanglement", methods=["POST"])
def request_entanglement():
    body = request.json or {}
    
    src, dst    = body.get("endpoint1"), body.get("endpoint2")
    mode        = body.get("mode","discrete").lower()
    
    try:
        overall_fidelity_threshold = float(body.get("f_final_th", 0.5))
        p_th                       = float(body.get("p_th",          0.9))
        f_th                       = float(body.get("f_th",          0.9))
    except (TypeError, ValueError):
        return jsonify({
            "error": "Invalid thresholds; p_th, f_th and f_final_th must be numbers"
        }), 400
    
    swap_penalty = -math.log(SWAP_EFFICIENCY * SWAP_SUCCESS_PROB)

    # find the optimal entanglement path
    node_path, state_path, phys_paths, costs = find_entanglement_path(
        BASE_GRAPH, src, dst, swap_penalty, p_th, f_th
    )
    if not node_path:
        return jsonify({"error":"No valid entanglement path"}),400

    # build the list of keys for the edges in the path
    edge_keys = [f"e{i}" for i in range(len(node_path)-1)]

    # build the work graph based on the found path and state transitions path
    # this graph will be used for the probabilistic simulation.
    work_graph = build_work_graph(
        BASE_GRAPH,
        node_path,
        phys_paths,
        state_path,
        edge_keys
    )

    # collapse the paths and states around SPDC nodes
    (
        collapsed_nodes,
        collapsed_edges,
        collapsed_phys_paths,
        collapsed_state
    ) = collapse_with_phys(
        node_path,    
        state_path,   
        edge_keys,   
        phys_paths    
    )

    # run the probabilistic simulation based on the mode
    if(mode == "discrete"):
        final_fid, memory, ops = simulate_discrete_prob(
            collapsed_nodes,
            collapsed_edges,
            work_graph
        )
    else:
        # Note: this is the more advanced simulator, which uses DEI swaps
        # and retries until success, so it can take longer.
        # It also returns a full memory of all link states.
        # The final_fid is the end-to-end fidelity of the entanglement.
        # The ops list contains all operations performed during the simulation.
        final_fid, memory, ops = simulate_continuous_prob(
            collapsed_nodes,
            collapsed_edges,
            work_graph,
            collapsed_state 
        )

    # check if the final fidelity is below the threshold
    if final_fid is None or final_fid < overall_fidelity_threshold:
        return jsonify({
            "error":"Entanglement failed",
            "logical_path": node_path,
            "state_path": state_path,
            "path": phys_paths,
            "hop_costs": costs,
            "memory": memory,
            "operations": ops,
            "final_fidelity": final_fid	
        }), 500

    # build the full physical chain from the collapsed paths containing all the actual physical nodes in the path
    full_physical_chain = build_full_physical_chain(collapsed_nodes, collapsed_phys_paths)
    
    # build the Qiskit circuit using the physical chain and operations list
    qasm, img_b64 = build_circuit_v2(full_physical_chain, ops)

    return jsonify({
        "status": "success",
        "logical_path":    node_path,
        "state_path":      state_path,
        "path":            full_physical_chain,
        "hop_costs":       costs,
        "memory":          memory,
        "operations":      ops,
        "final_fidelity":  final_fid,
        "circuit_qasm":    qasm,
        "circuit_img":     img_b64
    }), 200

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)