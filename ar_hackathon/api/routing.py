"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: Promit Das
Email address: promitdas345@gmail.com
*******************
"""

from typing import Optional, Dict, List, Tuple, Set
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

# ---------- internal helpers (no external libs) ----------

def _build_adjacency(state: GameState) -> Dict[str, List[Tuple[str, float, Optional[int]]]]:
    """
    Build an adjacency list from the current connections.
    Returns {from_fc: [(to_fc, weight, available_bandwidth), ...]}
    """
    adj: Dict[str, List[Tuple[str, float, Optional[int]]]] = {}
    for conn in state.connections:
        if conn.from_fc not in adj:
            adj[conn.from_fc] = []
        adj[conn.from_fc].append((conn.to_fc, conn.weight, conn.available_bandwidth))
    return adj


def _dijkstra(adj: Dict[str, List[Tuple[str, float, Optional[int]]]],
              source: str,
              destination: str,
              respect_bandwidth: bool = True) -> Optional[List[str]]:
    """
    Dijkstra's shortest path from source to destination.
    Returns the full path [source, ..., destination] or None if unreachable.
    
    When respect_bandwidth is True, edges with available_bandwidth <= 0 are skipped.
    """
    import heapq

    # dist[node] = shortest distance from source
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}
    # (distance, node)
    pq: List[Tuple[float, str]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == destination:
            break
        if d > dist.get(u, float('inf')):
            continue
        for (v, w, avail_bw) in adj.get(u, []):
            # Skip edges with no available bandwidth (Level 3)
            if respect_bandwidth and avail_bw is not None and avail_bw <= 0:
                continue
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Reconstruct path
    if destination not in prev:
        return None
    path = []
    cur: Optional[str] = destination
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def _dijkstra_next_hop(state: GameState, source: str, destination: str) -> Optional[str]:
    """
    Compute the next hop on the shortest path from source to destination.
    First tries respecting bandwidth constraints; falls back to ignoring them
    (the engine will simply reject the move if bandwidth is truly exhausted,
    and we'll retry next tick when bandwidth frees up).
    """
    adj = _build_adjacency(state)
    
    # Try with bandwidth awareness first
    path = _dijkstra(adj, source, destination, respect_bandwidth=True)
    
    if path is None or len(path) < 2:
        # Fall back: ignore bandwidth (wait-and-retry strategy)
        path = _dijkstra(adj, source, destination, respect_bandwidth=False)
    
    if path is not None and len(path) >= 2:
        return path[1]
    
    return None


# ---------- public API ----------

def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Determine the next FC to route a package to.
    
    Uses Dijkstra's algorithm to find the shortest weighted path from the
    package's current FC to its destination. For Level 3 (bandwidth-limited
    connections), edges with no remaining bandwidth are avoided when possible.
    
    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package
        
    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """
    # Already at destination (shouldn't happen, but be safe)
    if package.current_fc == package.destination_fc:
        return None

    return _dijkstra_next_hop(state, package.current_fc, package.destination_fc)
